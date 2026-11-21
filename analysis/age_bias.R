# Age-related bias in signed prediction errors.
# Input uses the same subject-level CSV as site_effects.R.

library(lmerTest)
library(emmeans)
library(dplyr)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop('Usage: Rscript analysis/age_bias.R <RESULTS_DIR>')
results_dir <- args[[1]]
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
input <- file.path(results_dir,
  'ART_pretraining_adaptation_subject_errors_mean_individual_errors.csv')
dat <- read_csv(input, show_col_types = FALSE)

required <- c('Subject', 'Pretraining', 'Adaptation', 'Age', 'Sex', 'signed_error')
missing <- setdiff(required, colnames(dat))
if (length(missing) > 0) stop(paste('Missing:', paste(missing, collapse = ', ')))

dat <- dat %>% mutate(
  Subject = factor(Subject), Sex = factor(Sex),
  Pretraining = factor(Pretraining,
    levels = c('TP1_White', 'TP2_Asian', 'TP2_Black', 'TP3')),
  Adaptation = factor(Adaptation,
    levels = c('Zero-shot', 'Traditional', 'LoRA', 'DoRA', 'PiSSA', 'FiltLoRA')),
  Age_c = Age - mean(Age, na.rm = TRUE)
)
contrasts(dat$Pretraining) <- contr.sum(4)
contrasts(dat$Adaptation) <- contr.sum(6)

fit <- lmer(signed_error ~ Age_c * Pretraining * Adaptation + Sex +
  (1 | Subject), data = dat, REML = FALSE)
write_csv(as.data.frame(anova(fit, type = 3)) %>%
  tibble::rownames_to_column('effect'),
  file.path(results_dir, 'AGE_omnibus_signed_error.csv'))

slopes <- emtrends(fit, ~ Pretraining * Adaptation, var = 'Age_c')
write_csv(as.data.frame(summary(slopes)),
  file.path(results_dir, 'AGE_slopes_pretraining_adaptation.csv'))

pretraining_slopes <- emtrends(fit, ~ Pretraining, var = 'Age_c')
adaptation_slopes <- emtrends(fit, ~ Adaptation, var = 'Age_c')
write_csv(as.data.frame(summary(pairs(pretraining_slopes, adjust = 'holm'))),
  file.path(results_dir, 'AGE_pretraining_contrasts.csv'))
write_csv(as.data.frame(summary(pairs(adaptation_slopes, adjust = 'holm'))),
  file.path(results_dir, 'AGE_adaptation_contrasts.csv'))
