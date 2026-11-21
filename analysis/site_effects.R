# =============================================================================
# PACKAGES
# =============================================================================

# install.packages(c("lmerTest", "emmeans", "dplyr", "readr", "tibble"))

library(lmerTest)
library(emmeans)
library(dplyr)
library(readr)
library(tibble)


# =============================================================================
# LOAD DATA
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop('Usage: Rscript analysis/site_effects.R <RESULTS_DIR>')
}
RESULTS_DIR <- args[[1]]
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

df <- read_csv(
  file.path(
    RESULTS_DIR,
    "ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv"
  ),
  show_col_types = FALSE
)


# =============================================================================
# FACTORS
# =============================================================================

df <- df %>%
  mutate(
    
    Subject = factor(Subject),
    
    Pretraining = factor(
      Pretraining,
      levels = c(
        "TP1_White",
        "TP2_Asian",
        "TP2_Black",
        "TP3"
      )
    ),
    
    Adaptation = factor(
      Adaptation,
      levels = c(
        "Zero-shot",
        "Traditional",
        "LoRA",
        "DoRA",
        "PiSSA",
        "FiltLoRA"
      )
    ),
    
    SiteGroup = factor(
      SiteGroup,
      levels = c(
        "Other sites",
        "Site F"
      )
    ),
    
    Sex = factor(
      Sex,
      levels = c(
        "Female",
        "Male"
      )
    ),
    
    # Center age for easier interpretation
    Age_c = Age - mean(Age)
  )


# Sum contrasts for Type III tests
contrasts(df$Pretraining) <- contr.sum(4)
contrasts(df$Adaptation)  <- contr.sum(6)
contrasts(df$SiteGroup)   <- contr.sum(2)
contrasts(df$Sex)         <- contr.sum(2)


# =============================================================================
# MODELS
# =============================================================================

m_site_abs <- lmer(
  abs_error ~
    Pretraining * Adaptation * SiteGroup +
    Age_c +
    Sex +
    (1 | Subject),
  data = df,
  REML = TRUE
)


m_site_signed <- lmer(
  signed_error ~
    Pretraining * Adaptation * SiteGroup +
    Age_c +
    Sex +
    (1 | Subject),
  data = df,
  REML = TRUE
)


# =============================================================================
# OMNIBUS TYPE III TESTS
# =============================================================================

anova_abs <- anova(
  m_site_abs,
  type = 3,
  ddf = "Satterthwaite"
)

anova_signed <- anova(
  m_site_signed,
  type = 3,
  ddf = "Satterthwaite"
)


cat("\n========================================\n")
cat("ABSOLUTE ERROR\n")
cat("========================================\n")
print(anova_abs)

cat("\n========================================\n")
cat("SIGNED ERROR\n")
cat("========================================\n")
print(anova_signed)


# =============================================================================
# KEEP ONLY SITE-RELATED TERMS
# =============================================================================

site_terms_abs <- anova_abs %>%
  as.data.frame() %>%
  rownames_to_column("Effect") %>%
  filter(
    grepl("SiteGroup", Effect)
  )


site_terms_signed <- anova_signed %>%
  as.data.frame() %>%
  rownames_to_column("Effect") %>%
  filter(
    grepl("SiteGroup", Effect)
  )


cat("\n========================================\n")
cat("SITE EFFECTS — ABSOLUTE ERROR\n")
cat("========================================\n")
print(site_terms_abs)

cat("\n========================================\n")
cat("SITE EFFECTS — SIGNED ERROR\n")
cat("========================================\n")
print(site_terms_signed)


# =============================================================================
# FUNCTION FOR SITE F vs OTHER SITES CONTRASTS
# =============================================================================

site_contrasts <- function(
    model,
    specs,
    by_vars = NULL
) {
  
  emm <- emmeans(
    model,
    specs = specs
  )
  
  # Because levels are:
  # Other sites, Site F
  #
  # revpairwise gives:
  # Site F - Other sites
  
  res <- contrast(
    emm,
    method = "revpairwise",
    by = by_vars,
    adjust = "none"
  )
  
  out <- as.data.frame(res) %>%
    rename(
      p_raw = p.value
    ) %>%
    mutate(
      p_holm = p.adjust(
        p_raw,
        method = "holm"
      ),
      
      significant = case_when(
        p_holm < 0.001 ~ "***",
        p_holm < 0.01  ~ "**",
        p_holm < 0.05  ~ "*",
        TRUE           ~ ""
      )
    )
  
  return(out)
}


# =============================================================================
# THREE-WAY FOLLOW-UP
#
# Use if:
# Pretraining × Adaptation × SiteGroup is significant.
#
# Site F vs Other sites in each of the 20 Pretraining × Adaptation cells.
# Holm across the 20 contrasts.
# =============================================================================

site_threeway_abs <- site_contrasts(
  m_site_abs,
  specs = ~ SiteGroup | Pretraining * Adaptation,
  by_vars = c(
    "Pretraining",
    "Adaptation"
  )
)

site_threeway_signed <- site_contrasts(
  m_site_signed,
  specs = ~ SiteGroup | Pretraining * Adaptation,
  by_vars = c(
    "Pretraining",
    "Adaptation"
  )
)


# =============================================================================
# ADAPTATION × SITE FOLLOW-UP
#
# Use if:
# Adaptation × SiteGroup is significant
# and the three-way interaction is not.
#
# 5 contrasts, Holm across 5.
# =============================================================================

site_adaptation_abs <- site_contrasts(
  m_site_abs,
  specs = ~ SiteGroup | Adaptation,
  by_vars = "Adaptation"
)

site_adaptation_signed <- site_contrasts(
  m_site_signed,
  specs = ~ SiteGroup | Adaptation,
  by_vars = "Adaptation"
)


# =============================================================================
# PRETRAINING × SITE FOLLOW-UP
#
# Use if:
# Pretraining × SiteGroup is significant
# and the three-way interaction is not.
#
# 4 contrasts, Holm across 4.
# =============================================================================

site_pretraining_abs <- site_contrasts(
  m_site_abs,
  specs = ~ SiteGroup | Pretraining,
  by_vars = "Pretraining"
)

site_pretraining_signed <- site_contrasts(
  m_site_signed,
  specs = ~ SiteGroup | Pretraining,
  by_vars = "Pretraining"
)


# =============================================================================
# PRINT POSSIBLE FOLLOW-UPS
# =============================================================================

cat("\n\n========================================\n")
cat("3-WAY FOLLOW-UP — ABSOLUTE ERROR\n")
cat("========================================\n")
print(site_threeway_abs)

cat("\n\n========================================\n")
cat("3-WAY FOLLOW-UP — SIGNED ERROR\n")
cat("========================================\n")
print(site_threeway_signed)


cat("\n\n========================================\n")
cat("ADAPTATION x SITE — ABSOLUTE ERROR\n")
cat("========================================\n")
print(site_adaptation_abs)

cat("\n\n========================================\n")
cat("ADAPTATION x SITE — SIGNED ERROR\n")
cat("========================================\n")
print(site_adaptation_signed)


cat("\n\n========================================\n")
cat("PRETRAINING x SITE — ABSOLUTE ERROR\n")
cat("========================================\n")
print(site_pretraining_abs)

cat("\n\n========================================\n")
cat("PRETRAINING x SITE — SIGNED ERROR\n")
cat("========================================\n")
print(site_pretraining_signed)


# =============================================================================
# SAVE
# =============================================================================

write_csv(
  site_terms_abs,
  file.path(
    RESULTS_DIR,
    "SITE_omnibus_absolute_error.csv"
  )
)

write_csv(
  site_terms_signed,
  file.path(
    RESULTS_DIR,
    "SITE_omnibus_signed_error.csv"
  )
)

write_csv(
  site_threeway_abs,
  file.path(
    RESULTS_DIR,
    "SITE_threeway_absolute_error.csv"
  )
)

write_csv(
  site_threeway_signed,
  file.path(
    RESULTS_DIR,
    "SITE_threeway_signed_error.csv"
  )
)

write_csv(
  site_adaptation_abs,
  file.path(
    RESULTS_DIR,
    "SITE_adaptation_absolute_error.csv"
  )
)

write_csv(
  site_adaptation_signed,
  file.path(
    RESULTS_DIR,
    "SITE_adaptation_signed_error.csv"
  )
)

write_csv(
  site_pretraining_abs,
  file.path(
    RESULTS_DIR,
    "SITE_pretraining_absolute_error.csv"
  )
)

write_csv(
  site_pretraining_signed,
  file.path(
    RESULTS_DIR,
    "SITE_pretraining_signed_error.csv"
  )
)

