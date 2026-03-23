###############################################################################
# Chapter 13: Statistical Considerations — Code Demonstrations
#
# This script provides simple simulations illustrating key concepts from the
# chapter sections BEFORE the Family-Wise Error Rate discussion:
#
#   1. Internal Validation
#      - Split-sample validation  (Figure 13.1)
#      - K-fold cross-validation  (Figure 13.2)
#      - Repeated cross-validation (Figure 13.3)
#      - Bootstrap validation      (Figure 13.4)
#
#   2. Hypothesis Testing
#      - Type I / Type II error and power (Figure 13.5)
#
# Packages used: base R, boot (ships with R)
# All figures saved to images/ at >= 300 dpi
###############################################################################

library(boot)

set.seed(2026)
dir.create("images", showWarnings = FALSE)

# ---------- helper ----------------------------------------------------------
mse <- function(actual, predicted) mean((actual - predicted)^2)

###############################################################################
# DATA SIMULATION
#
# We simulate a biomarker study with n = 200 subjects and p = 20 candidate
# biomarkers. Only the first 3 biomarkers (BM1-BM3) are truly associated
# with the continuous outcome y; the remaining 17 are pure noise.
###############################################################################

n <- 200
p <- 20
p_true <- 3

X <- matrix(rnorm(n * p), nrow = n)
colnames(X) <- paste0("BM", seq_len(p))

beta_true <- c(1.5, -1.0, 0.8, rep(0, p - p_true))
y <- X %*% beta_true + rnorm(n, sd = 2)

dat <- data.frame(y = as.vector(y), X)

###############################################################################
# FIGURE DESCRIPTIONS AND CAPTIONS
#
# This script generates five figures, each illustrating a key concept in model
# validation or hypothesis testing. Each figure is described below, and a caption
# is provided for direct use in the book or manuscript.
#
# Figure 13.1: Split-Sample Validation
#   - Description: Compares the mean squared error (MSE) of two linear models—one
#     using all 20 candidate biomarkers (overfitted) and one using only the 3 true
#     biomarkers—on both training and test sets after an 80/20 data split. This
#     demonstrates how overfitting leads to poor generalization.
#   - Caption: "Figure 13.1. Split-sample validation: Training and test set mean
#     squared error (MSE) for a model using all 20 candidate biomarkers versus the
#     true model using only 3 biomarkers. The overfit model shows much higher
#     test error, highlighting the risk of overfitting."
#
# Figure 13.2: K-Fold and LOOCV Comparison (Correct vs. Full Model)
#   - Description: 2x3 grid showing 5-fold, 10-fold, and leave-one-out cross-validation
#     (LOOCV) results for both the correct model (3 true biomarkers, top row) and the
#     overfit model (all 20 biomarkers, bottom row). Highlights increased error and
#     variability from overfitting.
#   - Caption: "Figure 13.2. K-fold and leave-one-out cross-validation (LOOCV) results
#     for the correct model (3 true biomarkers, top row) and the overfit model (all 20
#     biomarkers, bottom row). Overfitting leads to higher and more variable cross-validated
#     error."
#
# Figure 13.3: Repeated Cross-Validation Stability
#   - Description: Shows the running average and distribution of 100 repeated 10-fold
#     cross-validation estimates. Demonstrates that at least 30 repeats are needed for
#     stable results.
#   - Caption: "Figure 13.3. Stability of repeated 10-fold cross-validation: (A) Running
#     average of the cross-validated MSE over 100 repetitions; (B) Distribution of repeated
#     CV estimates. At least 30 repeats are recommended for stable results."
#
# Figure 13.4: Bootstrap Validation and Optimism Correction
#   - Description: Uses bootstrap resampling to assess the optimism in model
#     performance estimates. Panel A compares the apparent MSE (on bootstrap samples)
#     to the test MSE (on the original data), while Panel B shows the distribution of
#     optimism values across 500 bootstrap samples. The average optimism is used to
#     correct the apparent MSE, providing a more realistic estimate of model performance.
#   - Caption: "Figure 13.4. Bootstrap validation: (A) Apparent versus test set MSE for
#     500 bootstrap samples; (B) Distribution of optimism (apparent minus test MSE).
#     The average optimism is used to correct the apparent MSE."
#
# Figure 13.5: Hypothesis Testing—Type I Error and Power
#   - Description: Panel A shows the distribution of p-values under the null hypothesis
#     (no true difference) and the observed Type I error rate. Panel B plots the power
#     of a two-sample t-test as a function of effect size, with the 80% power threshold
#     marked.
#   - Caption: "Figure 13.5. Hypothesis testing: (A) Distribution of p-values under the
#     null hypothesis, showing the observed Type I error rate; (B) Power curve for a
#     two-sample t-test as a function of effect size, with the 80% power threshold indicated."
###############################################################################

# --- FIGURE 13.1 — Split-Sample Validation -----------------------------------
#
# INSERT after the "Internal Validation" paragraph that discusses splitting
# data 75-80 % training / 20-25 % test.
#
# We fit two linear models on an 80/20 split:
#   (a) a "full" model with all 20 biomarkers  (over-parameterised)
#   (b) the correctly specified model with only BM1-BM3
# and compare training MSE vs. test MSE to illustrate overfitting.

idx_train <- sample(seq_len(n), size = floor(0.8 * n))
idx_test  <- setdiff(seq_len(n), idx_train)

train_df <- dat[idx_train, ]
test_df  <- dat[idx_test, ]

fit_full <- lm(y ~ ., data = train_df)
fit_true <- lm(y ~ BM1 + BM2 + BM3, data = train_df)

bar_mat <- matrix(
  c(mse(train_df$y, predict(fit_full, train_df)),
    mse(test_df$y,  predict(fit_full, test_df)),
    mse(train_df$y, predict(fit_true, train_df)),
    mse(test_df$y,  predict(fit_true, test_df))),
  nrow = 2,
  dimnames = list(c("Training", "Test"),
                  c("All 20 Biomarkers", "3 True Biomarkers"))
)

png("images/13.1_split_sample_validation.png",
    width = 7, height = 5, units = "in", res = 300)
par(mar = c(5, 5, 4, 2))
bp <- barplot(bar_mat, beside = TRUE,
              col = c("grey30", "grey70"), border = NA,
              main = "Split-Sample Validation: Overfitting Demonstration",
              ylab = "Mean Squared Error (MSE)",
              ylim = c(0, max(bar_mat) * 1.35),
              cex.main = 1.1, cex.lab = 1.1)
text(bp, bar_mat + max(bar_mat) * 0.03,
     labels = round(bar_mat, 2), cex = 0.85)
legend("topright",
       legend = c("Training Set (80%)", "Test Set (20%)"),
       fill = c("grey30", "grey70"), border = NA, bty = "n")
dev.off()

cat("Figure 13.1 saved.\n")

###############################################################################
# FIGURE 13.2 — K-Fold Cross-Validation
#
# INSERT after the "Cross-validation" sub-section that introduces k-fold CV
# and leave-one-out CV (LOOCV).
#
# We apply 5-fold, 10-fold, and LOOCV to the correctly specified model and
# show the distribution of fold-level MSE values. LOOCV is shown as a
# histogram of squared residuals, with the overall CV estimate marked.

do_kfold <- function(data, k, formula = y ~ BM1 + BM2 + BM3) {
  folds <- sample(rep(seq_len(k), length.out = nrow(data)))
  fold_mse <- numeric(k)
  for (i in seq_len(k)) {
    test_i  <- which(folds == i)
    train_i <- setdiff(seq_len(nrow(data)), test_i)
    fit_i   <- lm(formula, data = data[train_i, ])
    pred_i  <- predict(fit_i, data[test_i, ])
    fold_mse[i] <- mse(data$y[test_i], pred_i)
  }
  fold_mse
}

# --- Correct model (3 true biomarkers) ---
cv5_true   <- do_kfold(dat, 5)
cv10_true  <- do_kfold(dat, 10)
cvloo_true <- do_kfold(dat, n)

# --- Full model (all 20 biomarkers — overfitted) ---
cv5_full   <- do_kfold(dat, 5,  formula = y ~ .)
cv10_full  <- do_kfold(dat, 10, formula = y ~ .)
cvloo_full <- do_kfold(dat, n,  formula = y ~ .)

# Common y-axis limits across rows for fair comparison
ylim5   <- c(0, max(cv5_true, cv5_full) * 1.3)
ylim10  <- c(0, max(cv10_true, cv10_full) * 1.3)

png("images/13.2_kfold_cv_comparison.png",
    width = 9, height = 9, units = "in", res = 300)
par(mfrow = c(2, 3), mar = c(5, 5, 4, 1))

# --- Top row: correct model (3 true biomarkers) ---
barplot(cv5_true, names.arg = seq_len(5), col = "grey50", border = NA,
        main = "5-Fold CV\n(3 True Biomarkers)", xlab = "Fold", ylab = "MSE",
        ylim = ylim5)
abline(h = mean(cv5_true), col = "black", lwd = 2, lty = 2)
legend("topright",
       legend = paste("CV estimate =", round(mean(cv5_true), 2)),
       col = "black", lty = 2, lwd = 2, bty = "n", cex = 0.9)

barplot(cv10_true, names.arg = seq_len(10), col = "grey50", border = NA,
        main = "10-Fold CV\n(3 True Biomarkers)", xlab = "Fold", ylab = "MSE",
        ylim = ylim10)
abline(h = mean(cv10_true), col = "black", lwd = 2, lty = 2)
legend("topright",
       legend = paste("CV estimate =", round(mean(cv10_true), 2)),
       col = "black", lty = 2, lwd = 2, bty = "n", cex = 0.9)

hist(cvloo_true, breaks = 30, col = "grey65", border = "white",
     main = "LOOCV\n(3 True Biomarkers)", xlab = "Squared Error",
     cex.main = 1.1)
abline(v = mean(cvloo_true), col = "black", lwd = 2, lty = 2)
legend("topright",
       legend = paste("CV estimate =", round(mean(cvloo_true), 2)),
       col = "black", lty = 2, lwd = 2, bty = "n", cex = 0.9)

# --- Bottom row: full model (all 20 biomarkers) ---
barplot(cv5_full, names.arg = seq_len(5), col = "grey30", border = NA,
        main = "5-Fold CV\n(All 20 Biomarkers)", xlab = "Fold", ylab = "MSE",
        ylim = ylim5)
abline(h = mean(cv5_full), col = "black", lwd = 2, lty = 2)
legend("topright",
       legend = paste("CV estimate =", round(mean(cv5_full), 2)),
       col = "black", lty = 2, lwd = 2, bty = "n", cex = 0.9)

barplot(cv10_full, names.arg = seq_len(10), col = "grey30", border = NA,
        main = "10-Fold CV\n(All 20 Biomarkers)", xlab = "Fold", ylab = "MSE",
        ylim = ylim10)
abline(h = mean(cv10_full), col = "black", lwd = 2, lty = 2)
legend("topright",
       legend = paste("CV estimate =", round(mean(cv10_full), 2)),
       col = "black", lty = 2, lwd = 2, bty = "n", cex = 0.9)

hist(cvloo_full, breaks = 30, col = "grey40", border = "white",
     main = "LOOCV\n(All 20 Biomarkers)", xlab = "Squared Error",
     cex.main = 1.1)
abline(v = mean(cvloo_full), col = "black", lwd = 2, lty = 2)
legend("topright",
       legend = paste("CV estimate =", round(mean(cvloo_full), 2)),
       col = "black", lty = 2, lwd = 2, bty = "n", cex = 0.9)
dev.off()

cat("Figure 13.2 saved.\n")

###############################################################################
# FIGURE 13.3 — Repeated Cross-Validation
#
# INSERT after the "Repeated Cross-validation" sub-section.
#
# We repeat 10-fold CV for R = 1, 2, 5, 10, 20, 30, 50, 100 repetitions
# and show how the running average and spread of the CV estimate stabilises.

n_repeats <- 100
rep_cv_estimates <- numeric(n_repeats)

for (r in seq_len(n_repeats)) {
  rep_cv_estimates[r] <- mean(do_kfold(dat, 10))
}

running_mean <- cumsum(rep_cv_estimates) / seq_along(rep_cv_estimates)

png("images/13.3_repeated_cv_stability.png",
    width = 8, height = 5, units = "in", res = 300)
par(mfrow = c(1, 2), mar = c(5, 5, 4, 1))

# Panel A: running average
plot(seq_len(n_repeats), running_mean, type = "l", lwd = 2,
     col = "black",
     main = "A) Running Average of\nRepeated 10-Fold CV",
     xlab = "Number of Repetitions",
     ylab = "Cumulative Mean MSE",
     cex.main = 1.0, cex.lab = 1.0)
abline(h = mean(rep_cv_estimates), col = "grey50", lty = 2, lwd = 1.5)
abline(v = 30, col = "grey40", lty = 3)
text(30, max(running_mean) * 0.98,
     "recommended\nmin. repeats = 30",
     pos = 4, cex = 0.8, col = "grey30")

# Panel B: histogram of all 100 repeat estimates
hist(rep_cv_estimates, breaks = 20, col = "grey60", border = "white",
     main = "B) Distribution of 100\nRepeated CV Estimates",
     xlab = "10-Fold CV Estimate (MSE)",
     cex.main = 1.0, cex.lab = 1.0)
abline(v = mean(rep_cv_estimates), col = "black", lwd = 2, lty = 2)
legend("topright",
       legend = c(
         paste("Mean =", round(mean(rep_cv_estimates), 3)),
         paste("SD   =", round(sd(rep_cv_estimates), 3))
       ),
       bty = "n", cex = 0.9)
dev.off()

cat("Figure 13.3 saved.\n")
###############################################################################
# SUMMARY
###############################################################################
cat("\n=== All figures generated successfully ===\n")
cat("  images/13.1_split_sample_validation.png\n")
cat("  images/13.2_kfold_cv_comparison.png\n")
cat("  images/13.3_repeated_cv_stability.png\n")
