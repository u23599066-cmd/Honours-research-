# ============================================================
# Monte Carlo Simulation: LASSO & Ridge
# ============================================================

rm(list = ls())
library(glmnet)
set.seed(123)

# Simulation Parameters
n_grid     <- c(20, 50, 100)
gamma_grid <- c(0, 0.5, 0.9, 0.99)
p          <- 10
R          <- 200
pi_thr     <- 0.8

beta <- matrix(c(0.5, 1, 2, 3, rep(0, 7)), nrow = p + 1)
feature_names <- c("X0", paste0("X", 1:p))

# Jaccard Similarity Function
compute_jaccard_fast <- function(selec_mat) {
  intersection <- selec_mat %*% t(selec_mat)
  sums <- rowSums(selec_mat)
  union <- outer(sums, sums, "+") - intersection
  jaccard_mat <- intersection / union
  jaccard_mat[is.nan(jaccard_mat)] <- 1
  return(mean(jaccard_mat[upper.tri(jaccard_mat)]))
}

# Main Simulation Engine
run_grid_simulation <- function(alpha_val, model_name) {
  
  jaccard_summary <- matrix(0, nrow = length(n_grid), ncol = length(gamma_grid),
                            dimnames = list(paste0("n=", n_grid), paste0("gamma=", gamma_grid)))
  
  # Storage for plotting & coefficient statistics
  prob_list <- list()
  conv_list <- list()
  stats_list <- list()
  
  for (i in seq_along(n_grid)) {
    for (j in seq_along(gamma_grid)) {
      
      n <- n_grid[i]
      gamma <- gamma_grid[j]
      setting_key <- paste0("n_", n, "_gamma_", gamma)
      
      coef_mat  <- matrix(0, nrow = R, ncol = p + 1)
      selec_mat <- matrix(0, nrow = R, ncol = p + 1)
      
      for (r in 1:R) {
        X_dum <- matrix(rnorm(n * (p + 1)), nrow = n, ncol = p + 1)
        Z     <- X_dum[, 1:p]
        z_p1  <- X_dum[, p + 1]
        
        X_features <- sqrt(1 - gamma^2) * Z + gamma * z_p1
        X          <- cbind(1, X_features)
        y          <- X %*% beta + rnorm(n)
        
        cv <- cv.glmnet(X, y, alpha = alpha_val, nfolds = 5, intercept = FALSE)
        model <- glmnet(X, y, alpha = alpha_val, lambda = cv$lambda.min, intercept = FALSE)
        
        coef_estimates <- as.numeric(coef(model)[-1])
        coef_mat[r, ]  <- coef_estimates
        
        # Non-zero tol
        selec_mat[r, ] <- as.numeric(abs(coef_estimates) > 1e-5)
      }
      
      # Metrics
      selection_prob <- colMeans(selec_mat)
      cum_mean       <- apply(selec_mat, 2, cumsum) / (1:R)
      mean_jaccard   <- compute_jaccard_fast(selec_mat)
      
      # Coefficient summary metrics
      beta_hat_mean <- colMeans(coef_mat)
      beta_hat_var  <- apply(coef_mat, 2, var)
      beta_hat_bias <- beta_hat_mean - as.numeric(beta)
      
      stats_df <- data.frame(
        E_beta_hat   = round(beta_hat_mean, 4),
        var_beta_hat = round(beta_hat_var, 4),
        bias         = round(beta_hat_bias, 4)
      )
      
      # Store Results
      jaccard_summary[i, j]    <- round(mean_jaccard, 4)
      prob_list[[setting_key]]  <- selection_prob
      conv_list[[setting_key]]  <- cum_mean
      stats_list[[setting_key]] <- stats_df
    }
  }
  
  # ------------------------------------------------------------
  # Plots
  # ------------------------------------------------------------
  
  # Plot 1: Feature Selection Probabilities Grid
  par(mfrow = c(3, 4), mar = c(3.5, 3.5, 2.5, 1), mgp = c(2, 0.7, 0))
  for (n in n_grid) {
    for (gamma in gamma_grid) {
      setting_key <- paste0("n_", n, "_gamma_", gamma)
      sp <- prob_list[[setting_key]]
      
      barplot(sp, names.arg = feature_names, col = "steelblue", las = 2,
              ylim = c(0, 1), main = paste0(model_name, " (n=", n, ", γ=", gamma, ")"),
              ylab = "Sel. Prob", xlab = "")
      abline(h = pi_thr, col = "red", lty = 2, lwd = 1.5)
    }
  }
  
  # Plot 2: Cumulative Selection Probability Convergence Grid
  par(mfrow = c(3, 4), mar = c(3.5, 3.5, 2.5, 1), mgp = c(2, 0.7, 0))
  color_palette <- if(alpha_val == 1) {
    c("black", "blue", "blue", "blue", "deeppink", "black", "green", rep("turquoise", 4))
  } else {
    c("black", rep("turquoise", p))
  }
  
  for (n in n_grid) {
    for (gamma in gamma_grid) {
      setting_key <- paste0("n_", n, "_gamma_", gamma)
      cm <- conv_list[[setting_key]]
      
      matplot(cm, type = "l", lty = 1, lwd = 1.5, col = color_palette,
              ylim = c(0, 1), xlab = "MC Run", ylab = "Cum. Prob",
              main = paste0(model_name, " (n=", n, ", γ=", gamma, ")"))
    }
  }
  
  # Reset graphics device layout
  par(mfrow = c(1, 1))
  
  return(list(
    jaccard = jaccard_summary,
    stats   = stats_list
  ))
}

# Helper function to generate combined parameter table for a specific (n, gamma)
generate_comparison_table <- function(lasso_stats, ridge_stats, n_val, gamma_val) {
  setting_key <- paste0("n_", n_val, "_gamma_", gamma_val)
  
  l_df <- lasso_stats[[setting_key]]
  r_df <- ridge_stats[[setting_key]]
  
  comp_df <- data.frame(
    Term         = c("X0 (Intercept)", paste0("X", 1:p)),
    True_beta    = round(as.numeric(beta), 4),
    LASSO_E_beta = l_df$E_beta_hat,
    LASSO_Var    = l_df$var_beta_hat,
    LASSO_Bias   = l_df$bias,
    Ridge_E_beta = r_df$E_beta_hat,
    Ridge_Var    = r_df$var_beta_hat,
    Ridge_Bias   = r_df$bias
  )
  
  colnames(comp_df) <- c("Term", "True β", "LASSO E(β_hat)", "LASSO Var(β_hat)", "LASSO Bias",
                         "Ridge E(β_hat)", "Ridge Var(β_hat)", "Ridge Bias")
  return(comp_df)
}

# ============================================================
# Run Simulations
# ============================================================

cat("\n================ RUNNING LASSO GRID SIMULATION ================\n")
lasso_res <- run_grid_simulation(alpha_val = 1, model_name = "LASSO")

cat("\n================ RUNNING RIDGE GRID SIMULATION ================\n")
ridge_res <- run_grid_simulation(alpha_val = 0, model_name = "Ridge")

# Extract Jaccard matrices
lasso_jaccard_table <- lasso_res$jaccard
ridge_jaccard_table <- ridge_res$jaccard

# Print Summary Tables
cat("\n================ MEAN JACCARD SIMILARITY INDEX ================\n\n")

cat("--- LASSO Model Mean Jaccard Indices ---\n")
print(lasso_jaccard_table)

cat("\n--- Ridge Model Mean Jaccard Indices ---\n")
print(ridge_jaccard_table)

# ============================================================
# Print Combined Parameter Estimation Tables for All Settings
# ============================================================

all_comparison_tables <- list()

for (n_val in n_grid) {
  for (gamma_val in gamma_grid) {
    setting_key <- paste0("n_", n_val, "_gamma_", gamma_val)
    
    cat("\n============================================================\n")
    cat("  Coefficient Statistics Comparison: n =", n_val, "| gamma =", gamma_val, "\n")
    cat("============================================================\n")
    
    comp_tbl <- generate_comparison_table(lasso_res$stats, ridge_res$stats, n_val, gamma_val)
    print(comp_tbl, row.names = FALSE)
    
    all_comparison_tables[[setting_key]] <- comp_tbl
  }
}

