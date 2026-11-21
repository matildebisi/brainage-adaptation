# =============================================================================
# PACKAGES
# =============================================================================

# install.packages(c("ARTool", "emmeans", "dplyr", "readr"))

library(ARTool)
library(emmeans)
library(dplyr)
library(readr)

emm_options(
  pbkrtest.limit = 5000,
  lmerTest.limit = 5000
)

# =============================================================================
# PATHS
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || length(args) > 3) {
  stop('Usage: Rscript analysis/improvement_analysis.R <RESULTS_DIR> <SPLIT_DIR> [SPLIT_PREFIX]')
}
RESULTS_DIR <- args[[1]]
SPLIT_DIR <- args[[2]]
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

CSV_PATH <- file.path(
  RESULTS_DIR,
  "ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv"
)


# =============================================================================
# LOAD DATA
# =============================================================================

df <- read_csv(
  CSV_PATH,
  show_col_types = FALSE
)

pretraining_levels <- c(
  "TP1_White",
  "TP2_Asian",
  "TP2_Black",
  "TP3"
)

adaptation_levels <- c(
  "Zero-shot",
  "Traditional",
  "LoRA",
  "DoRA",
  "PiSSA",
  "FiltLoRA"
)

df <- df %>%
  mutate(
    Subject = factor(Subject),
    
    Pretraining = factor(
      Pretraining,
      levels = pretraining_levels
    ),
    
    Adaptation = factor(
      Adaptation,
      levels = adaptation_levels
    )
  )


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

descriptive <- df %>%
  group_by(
    Pretraining,
    Adaptation
  ) %>%
  summarise(
    median_abs = median(abs_error),
    IQR_abs = IQR(abs_error),
    
    median_signed = median(signed_error),
    IQR_signed = IQR(signed_error),
    
    .groups = "drop"
  )


# =============================================================================
# RAW CHANGE RELATIVE TO ZERO-SHOT
#
# Useful for interpreting the ART results in years.
# =============================================================================

zero_shot <- df %>%
  filter(
    Adaptation == "Zero-shot"
  ) %>%
  select(
    Subject,
    Pretraining,
    zero_abs = abs_error,
    zero_signed = signed_error
  )


raw_deltas <- df %>%
  filter(
    Adaptation != "Zero-shot"
  ) %>%
  left_join(
    zero_shot,
    by = c(
      "Subject",
      "Pretraining"
    )
  ) %>%
  mutate(
    delta_abs =
      abs_error - zero_abs,
    
    delta_signed =
      signed_error - zero_signed
  )


delta_summary <- raw_deltas %>%
  group_by(
    Pretraining,
    Adaptation
  ) %>%
  summarise(
    median_delta_abs = median(delta_abs),
    IQR_delta_abs = IQR(delta_abs),
    
    median_delta_signed = median(delta_signed),
    IQR_delta_signed = IQR(delta_signed),
    
    .groups = "drop"
  )

# =============================================================================
# NAIVE MEAN-AGE BASELINE
#
# For each CV fold:
#   prediction = mean chronological age of that fold's TRAINING subjects
#
# The prediction is then applied to every held-out test subject.
# Absolute errors are computed separately for the four fold-specific
# mean predictors and averaged at subject level.
#
# This provides a naive target-domain reference baseline.
# =============================================================================

library(dplyr)
library(readr)
library(tidyr)
library(purrr)
library(tibble)


# -----------------------------------------------------------------------------
# PATH TO ORIGINAL 4-FOLD SPLITS
# -----------------------------------------------------------------------------

# Set from the second command-line argument.

SPLIT_PREFIX <- if (length(args) == 3) args[[3]] else "dataset"
N_FOLDS <- 4


# =============================================================================
# 1. COMPUTE TRAINING MEAN AGE FOR EACH FOLD
# =============================================================================

fold_mean_rows <- list()

for (fold in 1:N_FOLDS) {
  
  train_file <- file.path(
    SPLIT_DIR,
    paste0(
      SPLIT_PREFIX,
      "_fold_",
      fold,
      "_base_train_rows.csv"
    )
  )
  
  if (!file.exists(train_file)) {
    stop(
      paste0(
        "Missing training split file: ",
        train_file
      )
    )
  }
  
  train_df <- read_csv(
    train_file,
    show_col_types = FALSE
  )
  
  if (!"AGE" %in% names(train_df)) {
    stop(
      paste0(
        "AGE column missing from: ",
        train_file
      )
    )
  }
  
  # There should be 156 unique subjects in each training fold
  n_subjects <- n_distinct(train_df$SUBJ_ID)
  
  mean_age <- mean(
    train_df$AGE,
    na.rm = TRUE
  )
  
  fold_mean_rows[[fold]] <- tibble(
    Fold = fold,
    n_train = n_subjects,
    train_mean_age = mean_age
  )
}


fold_means <- bind_rows(
  fold_mean_rows
)


cat("\n========================================\n")
cat("TRAINING-SET MEAN AGE BY FOLD\n")
cat("========================================\n")

print(fold_means)


write_csv(
  fold_means,
  file.path(
    RESULTS_DIR,
    "mean_baseline_training_means.csv"
  )
)


# =============================================================================
# 2. GET THE TEST SUBJECTS USED IN THE IMPROVEMENT ANALYSIS
# =============================================================================

test_subjects <- df %>%
  
  distinct(
    Subject,
    Age
  )


if (nrow(test_subjects) != n_distinct(df$Subject)) {
  stop(
    "Each subject should have exactly one chronological age."
  )
}


cat("\nNumber of test subjects:", nrow(test_subjects), "\n")
# =============================================================================
# 3. GENERATE THE FOUR FOLD-SPECIFIC MEAN-AGE PREDICTIONS
# =============================================================================

mean_baseline_by_fold <- test_subjects %>%
  
  crossing(
    fold_means
  ) %>%
  
  mutate(
    
    predicted_age_mean_baseline =
      train_mean_age,
    
    signed_error_mean_baseline =
      predicted_age_mean_baseline - Age,
    
    abs_error_mean_baseline =
      abs(
        signed_error_mean_baseline
      )
  )


cat("\n========================================\n")
cat("MEAN BASELINE: FOLD-SPECIFIC TEST PERFORMANCE\n")
cat("========================================\n")


mean_baseline_fold_metrics <- mean_baseline_by_fold %>%
  
  group_by(
    Fold
  ) %>%
  
  summarise(
    
    n = n(),
    
    MAE = mean(
      abs_error_mean_baseline
    ),
    
    RMSE = sqrt(
      mean(
        signed_error_mean_baseline^2
      )
    ),
    
    BAG = mean(
      signed_error_mean_baseline
    ),
    
    .groups = "drop"
  )


print(
  mean_baseline_fold_metrics
)


write_csv(
  mean_baseline_fold_metrics,
  file.path(
    RESULTS_DIR,
    "mean_baseline_metrics_by_fold.csv"
  )
)

# =============================================================================
# 4. SUBJECT-LEVEL MEAN BASELINE ERROR
#
# This is the quantity used for paired comparisons with the model errors.
# =============================================================================

mean_baseline_subject <- mean_baseline_by_fold %>%
  
  group_by(
    Subject,
    Age
  ) %>%
  
  summarise(
    
    predicted_age_mean =
      mean(
        predicted_age_mean_baseline
      ),
    
    signed_error_mean =
      mean(
        signed_error_mean_baseline
      ),
    
    abs_error_mean =
      mean(
        abs_error_mean_baseline
      ),
    
    .groups = "drop"
  )


cat("\n========================================\n")
cat("MEAN BASELINE: SUBJECT-LEVEL SUMMARY\n")
cat("========================================\n")

cat(
  "Overall mean-baseline MAE:",
  mean(mean_baseline_subject$abs_error_mean),
  "\n"
)

cat(
  "Median absolute error:",
  median(mean_baseline_subject$abs_error_mean),
  "\n"
)


write_csv(
  mean_baseline_subject,
  file.path(
    RESULTS_DIR,
    "mean_baseline_subject_errors.csv"
  )
)

# =============================================================================
# 5. JOIN MEAN BASELINE TO MODEL ERRORS
# =============================================================================

comparison_df <- df %>%
  
  left_join(
    mean_baseline_subject %>%
      select(
        Subject,
        abs_error_mean
      ),
    by = "Subject"
  )


if (any(is.na(comparison_df$abs_error_mean))) {
  stop(
    "Some model subjects have no mean-baseline error."
  )
}


# =============================================================================
# 6. PAIRED WILCOXON:
#    EACH MODEL CONDITION VS MEAN-AGE BASELINE
#
# Negative delta:
#   model has LOWER absolute error than mean baseline.
#
# We use two-sided tests, consistent with the existing improvement analysis.
# =============================================================================

comparison_rows <- list()

counter <- 1


for (tp in levels(df$Pretraining)) {
  
  for (adapt in levels(df$Adaptation)) {
    
    sub <- comparison_df %>%
      
      filter(
        Pretraining == tp,
        Adaptation == adapt
      ) %>%
      
      arrange(
        Subject
      )
    
    
    if (nrow(sub) == 0) {
      next
    }
    
    
    model_error <- sub$abs_error
    mean_error  <- sub$abs_error_mean
    
    
    # Paired difference:
    # negative = model better than naive mean predictor
    delta <- model_error - mean_error
    
    
    test <- wilcox.test(
      model_error,
      mean_error,
      paired = TRUE,
      alternative = "two.sided",
      exact = FALSE
    )
    
    
    comparison_rows[[counter]] <- tibble(
      
      Pretraining = tp,
      Adaptation = adapt,
      
      n_subjects = nrow(sub),
      
      model_MAE =
        mean(
          model_error,
          na.rm = TRUE
        ),
      
      mean_baseline_MAE =
        mean(
          mean_error,
          na.rm = TRUE
        ),
      
      median_model_abs_error =
        median(
          model_error,
          na.rm = TRUE
        ),
      
      median_mean_abs_error =
        median(
          mean_error,
          na.rm = TRUE
        ),
      
      mean_delta =
        mean(
          delta,
          na.rm = TRUE
        ),
      
      median_delta =
        median(
          delta,
          na.rm = TRUE
        ),
      
      proportion_model_better =
        mean(
          model_error < mean_error,
          na.rm = TRUE
        ),
      
      Wilcoxon_V =
        unname(
          test$statistic
        ),
      
      p_raw =
        test$p.value
    )
    
    
    counter <- counter + 1
  }
}


mean_vs_models <- bind_rows(
  comparison_rows
)

# =============================================================================
# 7. HOLM CORRECTION ACROSS ALL 20 PLANNED COMPARISONS
# =============================================================================

mean_vs_models <- mean_vs_models %>%
  
  mutate(
    
    p_holm =
      p.adjust(
        p_raw,
        method = "holm"
      ),
    
    significant =
      case_when(
        p_holm < 0.001 ~ "***",
        p_holm < 0.01  ~ "**",
        p_holm < 0.05  ~ "*",
        TRUE           ~ ""
      ),
    
    direction =
      case_when(
        median_delta < 0 ~ "Model lower error",
        median_delta > 0 ~ "Mean baseline lower error",
        TRUE             ~ "No median difference"
      )
  ) %>%
  
  arrange(
    Pretraining,
    Adaptation
  )


cat("\n\n")
cat("============================================================\n")
cat("MODEL VS TRAINING-MEAN BASELINE\n")
cat("============================================================\n\n")

print(
  mean_vs_models,
  n = Inf
)


write_csv(
  mean_vs_models,
  file.path(
    RESULTS_DIR,
    "mean_baseline_vs_models_wilcoxon.csv"
  )
)


# =============================================================================
# ART MODELS
# =============================================================================

m_abs <- art(
  abs_error ~
    Pretraining * Adaptation +
    (1 | Subject),
  data = df
)

m_signed <- art(
  signed_error ~
    Pretraining * Adaptation +
    (1 | Subject),
  data = df
)


# =============================================================================
# OMNIBUS TESTS
#
# Pretraining
# Adaptation
# Pretraining × Adaptation
# =============================================================================

anova_abs <- anova(m_abs)

anova_signed <- anova(m_signed)


cat("\n========================================\n")
cat("ABSOLUTE ERROR — OMNIBUS ART\n")
cat("========================================\n")
print(anova_abs)

cat("\n========================================\n")
cat("SIGNED ERROR — OMNIBUS ART\n")
cat("========================================\n")
print(anova_signed)


# =============================================================================
# ALL PAIRWISE COMPARISONS WITHIN EACH PRETRAINED MODEL
#
# 5 conditions -> 10 pairwise comparisons per TP
# 4 TP -> 40 comparisons
#
# Holm correction across all 40 comparisons, separately for each outcome.
# =============================================================================

within_pretraining_posthoc <- function(
    art_model,
    outcome_name
) {
  
  # ART-C model for comparisons between factor-level combinations
  m_con <- artlm.con(
    art_model,
    "Pretraining:Adaptation"
  )
  
  emm <- emmeans(
    m_con,
    ~ PretrainingAdaptation
  )
  
  emm_df <- as.data.frame(emm)
  
  levs <- as.character(
    emm_df$PretrainingAdaptation
  )
  
  contrast_list <- list()
  
  
  # ---------------------------------------------------------------------------
  # All 10 pairwise comparisons among the 5 adaptation conditions
  # within each pretrained model
  # ---------------------------------------------------------------------------
  
  for (tp in pretraining_levels) {
    
    idx <- which(
      grepl(
        tp,
        levs,
        fixed = TRUE
      )
    )
    
    tp_levels <- levs[idx]
    
    pairs <- combn(
      seq_along(idx),
      2
    )
    
    
    for (j in seq_len(ncol(pairs))) {
      
      i1 <- idx[pairs[1, j]]
      i2 <- idx[pairs[2, j]]
      
      adapt1 <- sub(
        paste0("^", tp),
        "",
        levs[i1]
      )
      
      adapt2 <- sub(
        paste0("^", tp),
        "",
        levs[i2]
      )
      
      w <- rep(
        0,
        length(levs)
      )
      
      w[i1] <- 1
      w[i2] <- -1
      
      
      contrast_name <- paste0(
        tp,
        ": ",
        adapt1,
        " - ",
        adapt2
      )
      
      contrast_list[[contrast_name]] <- w
    }
  }
  
  
  # Raw p-values
  res <- contrast(
    emm,
    method = contrast_list,
    adjust = "none"
  )
  
  
  out <- as.data.frame(
    summary(
      res,
      infer = c(TRUE, TRUE)
    )
  ) %>%
    
    rename(
      p_raw = p.value
    ) %>%
    
    mutate(
      
      Outcome = outcome_name,
      
      # Holm across ALL 40 pairwise comparisons
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
      
    ) %>%
    
    select(
      Outcome,
      contrast,
      estimate,
      SE,
      df,
      t.ratio,
      p_raw,
      p_holm,
      significant
    )
  
  
  return(out)
}


# =============================================================================
# ABSOLUTE ERROR
# =============================================================================

within_abs <- within_pretraining_posthoc(
  m_abs,
  "Absolute error"
)

print(within_abs)


# =============================================================================
# SIGNED ERROR
# =============================================================================

within_signed <- within_pretraining_posthoc(
  m_signed,
  "Signed error"
)

print(within_signed)


# =============================================================================
# SAVE
# =============================================================================

write_csv(
  within_abs,
  file.path(
    RESULTS_DIR,
    "ART_all_pairwise_within_pretraining_absolute_error.csv"
  )
)

write_csv(
  within_signed,
  file.path(
    RESULTS_DIR,
    "ART_all_pairwise_within_pretraining_signed_error.csv"
  )
)


# =============================================================================
# INTERACTION POST-HOC
#
# Tests:
#
# [(Adaptation - Zero-shot)_TP_i]
#              -
# [(Adaptation - Zero-shot)_TP_j]
#
# For each adaptation:
#   Traditional
#   LoRA
#   DoRA
#   PiSSA
#
# 6 pairs of pretrained models × 4 adaptations = 24 tests
#
# Holm correction across the 24 tests for each outcome separately.
# =============================================================================

interaction_posthoc <- function(
    art_model,
    outcome_name
) {
  
  # For difference-of-differences use the original ART interaction model
  m_interaction <- artlm(
    art_model,
    "Pretraining:Adaptation"
  )
  
  
  emm <- emmeans(
    m_interaction,
    ~ Pretraining * Adaptation
  )
  
  
  res <- contrast(
    emm,
    
    interaction = list(
      Pretraining = "pairwise",
      Adaptation = "trt.vs.ctrl1"
    ),
    
    adjust = "none"
  )
  
  
  out <- as.data.frame(
    summary(
      res,
      infer = c(TRUE, TRUE)
    )
  ) %>%
    
    rename(
      p_raw = p.value
    ) %>%
    
    mutate(
      Outcome = outcome_name,
      
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


interaction_abs <- interaction_posthoc(
  m_abs,
  "Absolute error"
)

interaction_signed <- interaction_posthoc(
  m_signed,
  "Signed error"
)


cat("\n========================================\n")
cat("ABSOLUTE ERROR — INTERACTION POST-HOC\n")
cat("========================================\n")
print(interaction_abs)

cat("\n========================================\n")
cat("SIGNED ERROR — INTERACTION POST-HOC\n")
cat("========================================\n")
print(interaction_signed)


# =============================================================================
# SAVE RESULTS
# =============================================================================

write_csv(
  descriptive,
  file.path(
    RESULTS_DIR,
    "ART_descriptive_statistics.csv"
  )
)

write_csv(
  delta_summary,
  file.path(
    RESULTS_DIR,
    "ART_raw_delta_summary.csv"
  )
)

write_csv(
  as.data.frame(anova_abs),
  file.path(
    RESULTS_DIR,
    "ART_omnibus_absolute_error.csv"
  )
)

write_csv(
  as.data.frame(anova_signed),
  file.path(
    RESULTS_DIR,
    "ART_omnibus_signed_error.csv"
  )
)

write_csv(
  within_abs,
  file.path(
    RESULTS_DIR,
    "ART_within_pretraining_absolute_error.csv"
  )
)

write_csv(
  within_signed,
  file.path(
    RESULTS_DIR,
    "ART_within_pretraining_signed_error.csv"
  )
)

write_csv(
  interaction_abs,
  file.path(
    RESULTS_DIR,
    "ART_interaction_absolute_error.csv"
  )
)

write_csv(
  interaction_signed,
  file.path(
    RESULTS_DIR,
    "ART_interaction_signed_error.csv"
  )
)

