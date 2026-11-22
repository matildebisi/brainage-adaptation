# Improvement analysis: Pretraining x Adaptation
# Omnibus ART tests, within-pretraining ART-C contrasts, interaction contrasts,
# and descriptive changes relative to zero-shot.
# Usage: Rscript analysis/improvement_analysis.R <RESULTS_DIR>

library(ARTool)
library(emmeans)
library(dplyr)
library(readr)

emm_options(pbkrtest.limit = 5000, lmerTest.limit = 5000)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript analysis/improvement_analysis.R <RESULTS_DIR>")
}
results_dir <- args[[1]]

pretraining_levels <- c("TP1_White", "TP2_Asian", "TP2_Black", "TP3")
adaptation_levels <- c("Zero-shot", "Traditional", "LoRA", "DoRA", "PiSSA", "FiltLoRA")

df <- read_csv(
  file.path(results_dir, "ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    Subject = factor(Subject),
    Pretraining = factor(Pretraining, levels = pretraining_levels),
    Adaptation = factor(Adaptation, levels = adaptation_levels)
  )

save_result <- function(x, filename) {
  write_csv(x, file.path(results_dir, filename))
}

add_holm <- function(x, outcome) {
  x %>%
    rename(p_raw = p.value) %>%
    mutate(
      Outcome = outcome,
      p_holm = p.adjust(p_raw, method = "holm"),
      significant = case_when(
        p_holm < 0.001 ~ "***",
        p_holm < 0.01 ~ "**",
        p_holm < 0.05 ~ "*",
        TRUE ~ ""
      )
    )
}

# Descriptive statistics and paired changes (in years) from zero-shot.
descriptive <- df %>%
  group_by(Pretraining, Adaptation) %>%
  summarise(
    median_abs = median(abs_error), IQR_abs = IQR(abs_error),
    median_signed = median(signed_error), IQR_signed = IQR(signed_error),
    .groups = "drop"
  )

zero_shot <- df %>%
  filter(Adaptation == "Zero-shot") %>%
  select(Subject, Pretraining, zero_abs = abs_error, zero_signed = signed_error)

delta_summary <- df %>%
  filter(Adaptation != "Zero-shot") %>%
  left_join(zero_shot, by = c("Subject", "Pretraining")) %>%
  mutate(
    delta_abs = abs_error - zero_abs,
    delta_signed = signed_error - zero_signed
  ) %>%
  group_by(Pretraining, Adaptation) %>%
  summarise(
    median_delta_abs = median(delta_abs), IQR_delta_abs = IQR(delta_abs),
    median_delta_signed = median(delta_signed), IQR_delta_signed = IQR(delta_signed),
    .groups = "drop"
  )

save_result(descriptive, "ART_descriptive_statistics.csv")
save_result(delta_summary, "ART_raw_delta_summary.csv")

# ART models (repeated measurements by subject) and omnibus tests.
m_abs <- art(abs_error ~ Pretraining * Adaptation + (1 | Subject), data = df)
m_signed <- art(signed_error ~ Pretraining * Adaptation + (1 | Subject), data = df)

for (entry in list(
  list(model = m_abs, suffix = "absolute_error"),
  list(model = m_signed, suffix = "signed_error")
)) {
  omnibus <- as.data.frame(anova(entry$model))
  save_result(omnibus, paste0("ART_omnibus_", entry$suffix, ".csv"))
  print(omnibus)
}

# ART-C: all adaptation pairs within each pretrained model.
# 6 adaptation conditions -> 15 pairs per pretraining -> 60 contrasts per outcome.
within_pretraining_posthoc <- function(art_model, outcome) {
  m_con <- artlm.con(art_model, "Pretraining:Adaptation")
  emm <- emmeans(m_con, ~ PretrainingAdaptation)
  levs <- as.character(as.data.frame(emm)$PretrainingAdaptation)

  contrasts <- list()
  for (tp in pretraining_levels) {
    idx <- which(grepl(tp, levs, fixed = TRUE))
    if (length(idx) < 2) next

    pairs <- combn(idx, 2)
    for (j in seq_len(ncol(pairs))) {
      i1 <- pairs[1, j]
      i2 <- pairs[2, j]
      name <- paste0(
        tp, ": ", sub(paste0("^", tp), "", levs[i1]),
        " - ", sub(paste0("^", tp), "", levs[i2])
      )
      weights <- numeric(length(levs))
      weights[i1] <- 1
      weights[i2] <- -1
      contrasts[[name]] <- weights
    }
  }

  as.data.frame(summary(contrast(emm, method = contrasts, adjust = "none"),
                        infer = c(TRUE, TRUE))) %>%
    add_holm(outcome) %>%
    select(Outcome, contrast, estimate, SE, df, t.ratio,
           p_raw, p_holm, significant)
}

# Interaction: differences in adaptation-vs-zero-shot effects across pretrainings.
# 6 pairs of pretrainings x 5 non-zero-shot adaptations = 30 contrasts per outcome.
interaction_posthoc <- function(art_model, outcome) {
  emm <- emmeans(artlm(art_model, "Pretraining:Adaptation"),
                 ~ Pretraining * Adaptation)
  contrasts <- contrast(
    emm,
    interaction = list(Pretraining = "pairwise", Adaptation = "trt.vs.ctrl1"),
    adjust = "none"
  )
  as.data.frame(summary(contrasts, infer = c(TRUE, TRUE))) %>%
    add_holm(outcome)
}

for (entry in list(
  list(model = m_abs, outcome = "Absolute error", suffix = "absolute_error"),
  list(model = m_signed, outcome = "Signed error", suffix = "signed_error")
)) {
  within <- within_pretraining_posthoc(entry$model, entry$outcome)
  interaction <- interaction_posthoc(entry$model, entry$outcome)

  save_result(within, paste0("ART_within_pretraining_", entry$suffix, ".csv"))
  save_result(interaction, paste0("ART_interaction_", entry$suffix, ".csv"))
  print(within)
  print(interaction)
}