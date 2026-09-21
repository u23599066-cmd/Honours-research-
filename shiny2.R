# ============================================================
# Feature Selection Stability Shiny App
# ============================================================

library(shiny)
library(shinythemes)
library(shinydashboard)
library(shinyWidgets)
library(shinyjs)
library(glmnet)
library(corrplot)
library(ggplot2)
library(plotly)
library(dplyr)
library(tidyr)
library(DT)
library(viridis)
library(bslib)

# ============================================================
# Helpers
# ============================================================
feature_levels <- function(p) paste0("X", seq_len(p))
order_features <- function(x, p) factor(as.character(x), levels = feature_levels(p))
sort_features  <- function(f) f[order(as.numeric(sub("^X", "", f)))]
DEFAULT_BETAS  <- c(X1 = 0.5, X2 = 1, X3 = 2, X4 = 3)

# ============================================================
# UI
# ============================================================
ui <- fluidPage(
  useShinyjs(),
  theme = bs_theme(
    bg = "#0b0f1a", fg = "#e1e8f0",
    primary = "#00b4d8", secondary = "#0077b6",
    success = "#06d6a0", info = "#118ab2",
    warning = "#ffd166", danger = "#ef476f",
    base_font = font_google("Inter"),
    heading_font = font_google("Inter")
  ),
  tags$head(tags$style(HTML("
    .content-wrapper { background-color: #0f1a2e; }
    .box { background: rgba(26, 26, 46, 0.85) !important; border: 1px solid #2a3a5c !important; border-radius: 12px !important; box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important; }
    .btn-primary { background: linear-gradient(135deg, #00b4d8, #0077b6) !important; border: none !important; border-radius: 8px !important; }
    .nav-tabs > li > a { color: #e1e8f0 !important; }
    .nav-tabs > li.active > a { background: #00b4d8 !important; color: #fff !important; }
    .tab-content { padding: 20px; background: rgba(26, 26, 46, 0.5); border-radius: 0 0 12px 12px; }
    .shiny-output-error { color: #ef476f; }
    .value-box { background: rgba(0, 180, 216, 0.1); border: 1px solid #00b4d8; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .value-box .number { font-size: 2.2em; font-weight: bold; color: #00b4d8; word-wrap: break-word; }
    .value-box .label { font-size: 0.9em; color: #8892b0; }
    .sidebar-panel { background: rgba(26, 26, 46, 0.8); padding: 20px; border-radius: 12px; border: 1px solid #2a3a5c; }
    .engine-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }
    .engine-sim { background: #0077b6; color: white; }
    .beta-panel { background: rgba(6, 214, 160, 0.06); border: 1px dashed #2a3a5c; border-radius: 10px; padding: 10px; }
    .beta-preview { font-family: 'Courier New', monospace; font-size: 0.8em; color: #06d6a0; word-wrap: break-word; }
    .geom-note { background: rgba(0, 180, 216, 0.08); border-left: 4px solid #00b4d8; padding: 12px 15px; margin: 10px 0; border-radius: 0 8px 8px 0; color: #b0b8c8; font-size: 0.9em; }
    .geom-note b { color: #00b4d8; }
    .sparse-flag { color: #ef476f; font-weight: bold; }
    .dense-flag  { color: #06d6a0; font-weight: bold; }
  "))),
  
  # Header
  div(style = "background: linear-gradient(135deg, #0a1628, #1a2a4a); padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #2a3a6a;",
      fluidRow(
        column(9,
               h1(style = "color: #00b4d8; font-weight: 700; margin: 0;",
                  icon("server"), "Feature Selection Stability"),
               h4(style = "color: #8892b0; margin: 5px 0 0 0;",
                  "Monte Carlo Simulation Framework for LASSO vs Ridge Regularisation")),
        column(3, div(style = "text-align: right;",
                      h5(style = "color: #64ffda;", icon("code"), " Simulation Engine"),
                      p(style = "color: #8892b0; font-size: 0.8em;", "Department of Statistics | UP")))
      )),
  
  sidebarLayout(
    sidebarPanel(width = 3, class = "sidebar-panel",
                 h4(icon("sliders-h"), " Simulation Controls", style = "color: #00b4d8; font-weight: 600;"),
                 hr(style = "border-color: #2a3a5c;"),
                 h5(icon("database"), " Data Generation Parameters", style = "color: #ffd166;"),
                 numericInput("n_obs", "Observations (n)", value = 100, min = 50, max = 500, step = 10),
                 numericInput("p_features", "Features (p)", value = 10, min = 5, max = 50, step = 1),
                 sliderInput("gamma_mc", "Multicollinearity (γ)", value = 0.9, min = 0, max = 0.99, step = 0.05),
                 hr(style = "border-color: #2a3a5c;"),
                 h5(icon("bullseye"), " True Model ", style = "color: #ffd166;"),
                 p(style = "color: #8892b0; font-size: 0.8em;",
                   "Select the relevant features"),
                 selectizeInput("relevant_features", "Relevant Features",
                                choices = paste0("X", 1:10),
                                selected = c("X1", "X2", "X3", "X4"),
                                multiple = TRUE,
                                options = list(plugins = list("remove_button"))),
                 div(class = "beta-panel", uiOutput("beta_inputs")),
                 div(style = "margin-top: 8px;",
                     actionLink("reset_betas", "Reset to default signal (0.5, 1, 2, 3)",
                                style = "color: #00b4d8; font-size: 0.8em;")),
                 div(style = "margin-top: 10px;",
                     h6(icon("vector-square"), " True β vector", style = "color: #8892b0; margin-bottom: 4px;"),
                     uiOutput("beta_preview")),
                 hr(style = "border-color: #2a3a5c;"),
                 h5(icon("cog"), " Algorithm Parameters", style = "color: #00b4d8;"),
                 numericInput("r_runs", "Iterations / Runs (R)", value = 100, min = 20, max = 500, step = 10),
                 numericInput("nfolds", "CV Folds", value = 5, min = 3, max = 10, step = 1),
                 numericInput("pi_threshold", "Stability Threshold (π_thr)", value = 0.8, min = 0.5, max = 0.95, step = 0.05),
                 sliderInput("seed_val", "Random Seed", value = 123, min = 1, max = 999),
                 br(),
                 actionButton("run_pipeline", "Execute Simulation", icon = icon("play"),
                              class = "btn-primary btn-lg", style = "width: 100%;"),
                 div(style = "margin-top: 15px;",
                     progressBar("pipeline_progress", value = 0, display_pct = TRUE, status = "info")),
                 div(style = "margin-top: 15px; text-align: center;",
                     span("Simulation Engine Active", class = "engine-badge engine-sim"))
    ),
    
    mainPanel(width = 9,
              tabsetPanel(id = "mainTabs", type = "pills",
                          
                          # ============================================================
                          # Tab 1: Pipeline Dashboard
                          # ============================================================
                          tabPanel(icon = icon("microchip"), "Pipeline Dashboard", br(),
                                   
                                   fluidRow(
                                     column(6, div(class = "value-box",
                                                   h5(class = "label", "Response Variable / Model"),
                                                   h4(class = "number", style = "font-size: 1.5em; color: #06d6a0;",
                                                      "Y ~ Xβ + ε (Gaussian)"))),
                                     column(6, div(class = "value-box",
                                                   h5(class = "label", "Active Signal (non-zero β)"),
                                                   htmlOutput("signal_summary")))),
                                   div(class = "box", style = "padding: 15px; margin-top: 15px;",
                                       h4(icon("clipboard"), " Pipeline Execution Log", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       verbatimTextOutput("pipeline_log", placeholder = TRUE)),
                                   div(class = "box", style = "padding: 15px; margin-top: 15px;",
                                       h4(icon("info-circle"), " Simulation Configuration Summary", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       DTOutput("engine_summary_table")),
                                   div(class = "box", style = "padding: 15px; margin-top: 15px;",
                                       h4(icon("bullseye"), " True Coefficients", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       plotlyOutput("true_beta_plot", height = "320px"))
                          ),
                          
                          # ============================================================
                          # Tab 2: LASSO Analysis
                          # ============================================================
                          tabPanel(icon = icon("chart-bar"), "LASSO Analysis", br(),
                                   div(class = "box", style = "padding: 15px;",
                                       h4(icon("chart-bar"), " LASSO Performance", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       fluidRow(
                                         column(6, plotlyOutput("lasso_selection_prob", height = "400px")),
                                         column(6, plotlyOutput("lasso_convergence", height = "400px"))),
                                       br(),
                                       fluidRow(column(12, plotlyOutput("lasso_coefficients", height = "400px")))),
                                   br(),
                                   fluidRow(
                                     column(6, div(class = "box", style = "padding: 15px;",
                                                   h4(icon("table"), " Coefficient Statistics", style = "color: #00b4d8;"),
                                                   hr(style = "border-color: #2a3a5c;"),
                                                   DTOutput("lasso_coef_table"))),
                                     column(6, div(class = "box", style = "padding: 15px;",
                                                   h4(icon("check-circle"), " Selection Stability Profile", style = "color: #00b4d8;"),
                                                   hr(style = "border-color: #2a3a5c;"),
                                                   fluidRow(
                                                     column(6, div(class = "value-box",
                                                                   h5(class = "label", "Mean Jaccard Similarity"),
                                                                   htmlOutput("lasso_jaccard"))),
                                                     column(6, div(class = "value-box",
                                                                   h5(class = "label", "Stable Selected Features"),
                                                                   htmlOutput("lasso_important")))),
                                                   div(style = "margin-top: 15px;",
                                                       h5(icon("list"), " Feature Selection Probabilities", style = "color: #8892b0;"),
                                                       verbatimTextOutput("lasso_probs")))))
                          ),
                          
                          # ============================================================
                          # Tab 3: Ridge Analysis
                          # ============================================================
                          tabPanel(icon = icon("chart-line"), "Ridge Analysis", br(),
                                   div(class = "box", style = "padding: 15px;",
                                       h4(icon("chart-line"), " Ridge Performance", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       fluidRow(
                                         column(6, plotlyOutput("ridge_selection_prob", height = "400px")),
                                         column(6, plotlyOutput("ridge_convergence", height = "400px"))),
                                       br(),
                                       fluidRow(column(12, plotlyOutput("ridge_coefficients", height = "400px")))),
                                   br(),
                                   fluidRow(
                                     column(6, div(class = "box", style = "padding: 15px;",
                                                   h4(icon("table"), " Coefficient Statistics", style = "color: #00b4d8;"),
                                                   hr(style = "border-color: #2a3a5c;"),
                                                   DTOutput("ridge_coef_table"))),
                                     column(6, div(class = "box", style = "padding: 15px;",
                                                   h4(icon("check-circle"), " Selection Stability Profile", style = "color: #00b4d8;"),
                                                   hr(style = "border-color: #2a3a5c;"),
                                                   fluidRow(
                                                     column(6, div(class = "value-box",
                                                                   h5(class = "label", "Mean Jaccard Similarity"),
                                                                   htmlOutput("ridge_jaccard"))),
                                                     column(6, div(class = "value-box",
                                                                   h5(class = "label", "Stable Predictor Count"),
                                                                   htmlOutput("ridge_important")))),
                                                   div(style = "margin-top: 15px;",
                                                       h5(icon("list"), " Ridge Inclusion Probabilities", style = "color: #8892b0;"),
                                                       verbatimTextOutput("ridge_probs")))))
                          ),
                          
                          # ============================================================
                          # Tab 4: LASSO vs Ridge
                          # ============================================================
                          tabPanel(icon = icon("balance-scale"), "LASSO vs Ridge", br(),
                                   div(class = "box", style = "padding: 15px;",
                                       h4(icon("balance-scale"), " Comparative Analysis View", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       fluidRow(
                                         column(6, plotlyOutput("compare_selection", height = "400px")),
                                         column(6, plotlyOutput("compare_jaccard", height = "400px"))),
                                       br(),
                                       fluidRow(column(12, plotlyOutput("compare_bias", height = "400px")))),
                                   br(),
                                   fluidRow(column(12, div(class = "box", style = "padding: 15px;",
                                                           h4(icon("clipboard-check"), " Performance Dashboard Summary", style = "color: #00b4d8;"),
                                                           hr(style = "border-color: #2a3a5c;"),
                                                           DTOutput("comparison_table"))))
                          ),
                          
                          # ============================================================
                          # Tab 5: 2D Geometric View
                          # ============================================================
                          tabPanel(icon = icon("shapes"), "2D Geometric View", br(),
                                   
                                   div(class = "geom-note",
                                       HTML("<b>How to read this:</b> The ellipses are RSS contours centred at the OLS estimate.
              The constraint region (circle for Ridge, diamond for LASSO) is where the penalised solution must live.
              As the ellipse grows from the OLS centre, it first touches the constraint boundary at the tangency point and this is where we find the penalised solution.
              <br><br>
              <b style='color:#00b4d8;'>LASSO:</b> The tangency often lands exactly on a diamond corner
              (an axis) → one or more coefficients become exactly zero → <b>feature selection</b>.
              <br>
              <b style='color:#ffd166;'>Ridge:</b> The circle is smooth, so the tangency lands <i>near</i> but
              rarely <i>on</i> an axis → coefficients are only shrunk, never exactly zero → <b>no feature selection</b>.")),
                                   
                                   div(class = "box", style = "padding: 15px; margin-top: 15px;",
                                       h4(icon("sliders-h"), " Geometric Diagram Controls", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       fluidRow(
                                         column(3, numericInput("geom_b1_ols", "OLS β̂₁", value = 3.0, min = -6, max = 6, step = 0.1)),
                                         column(3, numericInput("geom_b2_ols", "OLS β̂₂", value = 0.5, min = -6, max = 6, step = 0.1)),
                                         column(3, sliderInput("geom_t", "Constraint radius (t)", value = 1.5, min = 0.3, max = 6, step = 0.1)),
                                         column(3, sliderInput("geom_rho", "Predictor correlation (ρ)", value = 0.6, min = 0, max = 0.95, step = 0.05))),
                                       fluidRow(
                                         column(4, selectInput("geom_display", "Display Mode",
                                                               choices = c("Ridge & LASSO (side by side)",
                                                                           "Ridge only", "LASSO only"),
                                                               selected = "Ridge & LASSO (side by side)")),
                                         column(4, checkboxInput("geom_show_ellipses", "Show growing RSS ellipses", value = TRUE)),
                                         column(4, checkboxInput("geom_show_guides", "Show axis guide-lines to tangency", value = TRUE))),
                                       fluidRow(
                                         column(4, checkboxInput("geom_show_path", "Show OLS → solution shrinkage path", value = TRUE)),
                                         column(4, checkboxInput("geom_show_corner", "Highlight diamond corner if hit", value = TRUE)),
                                         column(4, div(style = "margin-top: 25px;",
                                                       actionButton("geom_auto_sparse", "Find sparse example",
                                                                    icon = icon("magic"),
                                                                    style = "width:100%; background:#0077b6; color:white; border:none; border-radius:8px;"))))
                                   ),
                                   
                                   div(class = "box", style = "padding: 15px; margin-top: 15px;",
                                       h4(icon("shapes"), " RSS Contours × Constraint Intersection", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       uiOutput("geom_2d_container")),
                                   
                                   br(),
                                   
                                   fluidRow(
                                     column(6, div(class = "box", style = "padding: 15px;",
                                                   h4(icon("circle-dot"), " Ridge Geometry (L2)", style = "color: #ffd166;"),
                                                   hr(style = "border-color: #2a3a5c;"),
                                                   div(class = "value-box",
                                                       h5(class = "label", "Ridge solution (β̂₁, β̂₂)"),
                                                       h4(class = "number", style = "font-size:1.4em; color:#ffd166;",
                                                          htmlOutput("geom_ridge_solution"))),
                                                   div(class = "value-box",
                                                       h5(class = "label", "Tangency location"),
                                                       htmlOutput("geom_ridge_tangency")),
                                                   p(style = "color:#8892b0; font-size:0.85em;",
                                                     "l2 constraint has no corners so the RSS ellipse touches it at a smooth point that
                            is not on the axis. Both coefficients remain non-zero — only shrunk."))),
                                     column(6, div(class = "box", style = "padding: 15px;",
                                                   h4(icon("diamond"), " LASSO Geometry (L1)", style = "color: #00b4d8;"),
                                                   hr(style = "border-color: #2a3a5c;"),
                                                   div(class = "value-box",
                                                       h5(class = "label", "LASSO solution (β̂₁, β̂₂)"),
                                                       h4(class = "number", style = "font-size:1.4em; color:#00b4d8;",
                                                          htmlOutput("geom_lasso_solution"))),
                                                   div(class = "value-box",
                                                       h5(class = "label", "Tangency location"),
                                                       htmlOutput("geom_lasso_tangency")),
                                                   p(style = "color:#8892b0; font-size:0.85em;",
                                                     "l1 constraint has corners on the axes. The growing ellipse reaches a corner first then
                            the corresponding coefficient is set exactly to zero (feature selection).")))
                                   ),
                                   
                                   div(class = "box", style = "padding: 15px;",
                                       h4(icon("square-root-variable"), " Formulation", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       fluidRow(
                                         column(6, div(class = "value-box",
                                                       h5(class = "label", "Ridge (L2) — circular constraint"),
                                                       h4(class = "number", style = "font-size: 1.3em; color:#ffd166;",
                                                          HTML("β₁² + β₂² ≤ t²")))),
                                         column(6, div(class = "value-box",
                                                       h5(class = "label", "LASSO (L1) — diamond constraint"),
                                                       h4(class = "number", style = "font-size: 1.3em; color:#00b4d8;",
                                                          HTML("|β₁| + |β₂| ≤ t"))))
                                       ))
                          ),
                          
                          # ============================================================
                          # Tab 6: Multicollinearity
                          # ============================================================
                          tabPanel(icon = icon("project-diagram"), "Multicollinearity", br(),
                                   div(class = "box", style = "padding: 15px;",
                                       h4(icon("project-diagram"), " Correlation Architecture", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       fluidRow(
                                         column(7, plotOutput("correlation_plot", height = "550px")),
                                         column(5, div(style = "padding: 10px;",
                                                       h5(icon("info"), " Multicollinearity Diagnostic Summary", style = "color: #00b4d8;"),
                                                       hr(style = "border-color: #2a3a5c;"),
                                                       div(class = "value-box", h5(class = "label", "Observations (n)"),
                                                           h4(class = "number", style = "font-size:1.8em;", htmlOutput("obs_display"))),
                                                       div(class = "value-box", h5(class = "label", "Features (p)"),
                                                           h4(class = "number", style = "font-size:1.8em;", htmlOutput("features_display"))),
                                                       div(class = "value-box", h5(class = "label", "Mean Absolute Correlation"),
                                                           h4(class = "number", style = "font-size:1.8em;", htmlOutput("mean_corr"))),
                                                       div(class = "value-box", h5(class = "label", "Condition Number (κ)"),
                                                           h4(class = "number", style = "font-size:1.8em;", htmlOutput("condition_num")))))))
                          ),
                          
                          # ============================================================
                          # Tab 7: Data Matrix
                          # ============================================================
                          tabPanel(icon = icon("table"), "Data Matrix", br(),
                                   div(class = "box", style = "padding: 15px;",
                                       h4(icon("table"), " Data Explorer", style = "color: #00b4d8;"),
                                       hr(style = "border-color: #2a3a5c;"),
                                       fluidRow(
                                         column(4, numericInput("show_rows", "Rows to Display", value = 20, min = 5, max = 100)),
                                         column(4, selectInput("matrix_type", "Matrix Type",
                                                               choices = c("Design Matrix (X)", "Correlation Matrix",
                                                                           "Selection Matrix", "True Coefficients (β)"))),
                                         column(4, actionButton("refresh_data", "Refresh Data", icon = icon("sync"),
                                                                class = "btn-primary", style = "margin-top: 25px; width: 100%;"))),
                                       br(), DTOutput("data_matrix_table"),
                                       div(style = "margin-top: 15px;",
                                           h5(icon("info"), " Matrix Dimensions", style = "color: #8892b0;"),
                                           verbatimTextOutput("matrix_dims")))
                          ),
                          
                          # ============================================================
                          # Tab 8: About
                          # ============================================================
                          tabPanel(icon = icon("info-circle"), "About", br(),
                                   fluidRow(column(12,
                                                   div(class = "box", style = "padding: 30px;",
                                                       h4(icon("graduation-cap"), " Research Overview", style = "color: #00b4d8;"),
                                                       hr(style = "border-color: #2a3a5c;"),
                                                       
                                                       div(style = "padding: 20px;",
                                                           h5(icon("book"), " Shrinkage and Sparsity in Feature Selection Stability in Simulation Studies",
                                                              style = "color: #64ffda;"),
                                                           br(),
                                                           p("This application accompanies the research report Shrinkage and Sparsity in Feature Selection Stability in Simulation Studies. It is an interactive
                                               simulation study, allowing the user to reproduce and explore the stability of LASSO and Ridge regularisation
                                               under varying sample sizes, multicollinearity levels and signal structures."),
                                                           
                                                           br(),
                                                           h5(icon("bullseye"), " Objective", style = "color: #64ffda;"),
                                                           p("The study evaluates ", strong("feature selection consistency"),
                                                             ", and assesses whether shrinkage-based (Ridge) and sparsity-based (LASSO) regularisation techniques
                                               yield consistent conclusions about feature importance. The app does this by running a
                                               Monte Carlo simulation and quantifying how stably each method recovers the true sparse model."),
                                                           p("Stability is measured
                                               here using the ", strong("selection probability"), " and the ", strong("Jaccard similarity index"),
                                                             "."),
                                                           
                                                           br(),
                                                           h5(icon("scale-unbalanced"), " Why Regularisation?", style = "color: #64ffda;"),
                                                           p("In high-dimensional settings the OLS estimator becomes unstable: X'X becomes ill-conditioned, coefficient
                                               variances inflate, and the CLRM assumptions are frequently violated. Regularisation trades a small amount of bias for a large
                                               reduction in variance — producing simpler, more
                                               generalisable and parsimonious models."),
                                                           
                                                           br(),
                                                           h5(icon("flask"), " Simulation Framework", style = "color: #64ffda;"),
                                                           p("A design matrix X of dimension n × p is generated. Multicollinearity is induced following the framework of du Plessis et al:"),
                                                           div(style = "text-align:center; margin: 12px 0;",
                                                               HTML("<code style='color:#00b4d8; background:#0f1a2e; padding:6px 12px; border-radius:6px;'>
                                                      xᵢⱼ = √(1 − γ²)·zᵢⱼ + γ·zᵢ(p+1),  zᵢⱼ ~ N(0, 1)</code>")),
                                                           p("where γ ∈ [0, 0.99] controls the degree of multicollinearity, giving Corr(Xⱼ, Xₖ) = γ² for j ≠ k.
                                               The response is generated as ", strong("y = Xβ + ε"), " with ε ~ N(0, σ²I), using a user-specified
                                               coefficient vector β."),
                                                           p("For each Monte Carlo run, LASSO (α = 1) and Ridge (α = 0) are fitted via ", strong("glmnet"),
                                                             " with λ chosen by ", strong("5-fold cross-validation"), ". A binary selection matrix is constructed,
                                               and stability metrics are computed across all R replications."),
                                                           
                                                           br(),
                                                           h5(icon("chart-line"), " Key Findings", style = "color: #64ffda;"),
                                                           tags$ul(
                                                             tags$li(strong("LASSO performs feature selection:"), " selection probabilities for the truly relevant
                                                       features clearly separate from noise features, with separation improving as n grows."),
                                                             tags$li(strong("Ridge never selects:"), " selection probabilities approach 1 for ",
                                                                     em("all"), " features regardless of relevance, and its Jaccard index is trivially 1 because the
                                                       selected set is always identical. Selection stability is therefore uninformative for Ridge."),
                                                             tags$li(strong("Sample size drives stability:"), " at γ = 0.99, only n = 100 exceeds the 0.8 threshold for all
                                                       relevant features. This is a clear indication that at high levels of multicollinearity a sufficient sample is required to acheive stability."),
                                                             tags$li(strong("Multicollinearity degrades stability:"), " as γ → 0.99, LASSO's mean Jaccard drops sharply
                                                       because LASSO arbitrarily selects one
                                                       feature from a highly correlated group and disregards the rest.")
                                                           ),
                                                           
                                                           br(),
                                                           h5(icon("user"), " Author", style = "color: #64ffda;"),
                                                           p("Vuledzani Mutshinyalo"),
                                                           p("Student Number: 23599066"),
                                                           
                                                           br(),
                                                           h5(icon("users"), " Supervisors", style = "color: #64ffda;"),
                                                           p("Dr J. Kleyn (Supervisor)"),
                                                           p("Prof S.M. Millard (Co-supervisor)"),
                                                           
                                                           br(),
                                                           h5(icon("university"), " Institution", style = "color: #64ffda;"),
                                                           p("Department of Statistics"),
                                                           p("University of Pretoria"),
                                                           
                                                           br(),
                                                           h5(icon("calendar"), " Report Date", style = "color: #64ffda;"),
                                                           p("October 2026"),
                                                           
                                                           br(),
                                                           h5(icon("code"), " Methodology Summary", style = "color: #64ffda;"),
                                                           tags$ul(
                                                             tags$li("Monte Carlo simulation with controlled multicollinearity"),
                                                             tags$li("LASSO and Ridge fitted via glmnet"),
                                                             tags$li("λ selected by k-fold cross-validation"),
                                                             tags$li("Binary feature-selection matrix"),
                                                             tags$li("Stability metrics: selection probability (π̂) and Jaccard similarity index (J)"),
                                                             tags$li("Selection threshold: π_thr (0.8)"),
                                                             tags$li("Estimator diagnostics: E(β̂), Var(β̂), Bias across R replications"),
                                                             tags$li("2D geometric interpretation of ℓ₁/ℓ₂ constraint regions and RSS tangency")
                                                           ),
                                                           
                                                           br(),
                                                           h5(icon("book-open"), " Key References", style = "color: #64ffda;"),
                                                           tags$ul(style = "font-size: 0.9em;",
                                                                   tags$li("Tibshirani, R. (1996). Regression shrinkage and selection via the LASSO. ",
                                                                           em("JRSS-B"), ", 58(1), 267–288."),
                                                                   tags$li("Hoerl, A.E. & Kennard, R.W. (1970). Ridge regression: biased estimation for nonorthogonal problems. ",
                                                                           em("Technometrics"), ", 12(1), 55–67."),
                                                                   tags$li("Meinshausen, N. & Bühlmann, P. (2010). Stability selection. ",
                                                                           em("JRSS-B"), ", 72(4), 417–473."),
                                                                   tags$li("Hastie, T., Tibshirani, R. & Friedman, J. (2009). ",
                                                                           em("The Elements of Statistical Learning"), ", 2nd ed. Springer."),
                                                                   tags$li("du Plessis, S., Arashi, M., Millard, S. & Maribe, G. (2026). Navigating multicollinearity in
                                                       linear regression models. ", em("Contemporary Mathematics"), ", 7(3).")
                                                           ),
                                                           
                                                           br(),
                                                           h5(icon("rocket"), " Engine Mode", style = "color: #64ffda;"),
                                                           tags$ul(tags$li(span("Simulation Mode", class = "engine-badge engine-sim"),
                                                                           " — Synthetic data with controlled multicollinearity"))
                                                       )
                                                   )
                                   ))
                          )
              )
    )
  )
)

# ============================================================
# Server
# ============================================================
server <- function(input, output, session) {
  
  results <- reactiveValues(
    lasso = NULL, ridge = NULL, cor_matrix = NULL, running = FALSE,
    log = "System initialization complete. Awaiting simulation execution call...",
    feature_names = NULL, beta_true = NULL, X_data = NULL, y_data = NULL, p_used = NULL
  )
  
  current_meta <- reactive({
    list(n = input$n_obs, p = input$p_features,
         features = feature_levels(input$p_features),
         mode_label = "Simulation (Synthetic)")
  })
  run_meta <- reactive({ req(results$p_used); list(p = results$p_used, features = feature_levels(results$p_used)) })
  
  # ---------- Signal specification ----------
  observeEvent(input$p_features, {
    req(input$p_features)
    fnames <- feature_levels(input$p_features)
    keep <- intersect(input$relevant_features, fnames)
    if (length(keep) == 0) keep <- intersect(names(DEFAULT_BETAS), fnames)
    updateSelectizeInput(session, "relevant_features", choices = fnames, selected = sort_features(keep))
  }, ignoreInit = TRUE)
  
  output$beta_inputs <- renderUI({
    sel <- input$relevant_features
    if (is.null(sel) || length(sel) == 0) {
      return(p(style = "color: #ef476f; font-size: 0.85em; margin: 0;",
               "No relevant features selected — the true model is pure noise (β = 0 for all)."))
    }
    sel <- sort_features(sel)
    lapply(sel, function(f) {
      val <- isolate(input[[paste0("beta_", f)]])
      if (is.null(val) || is.na(val)) val <- if (f %in% names(DEFAULT_BETAS)) unname(DEFAULT_BETAS[f]) else 1
      numericInput(paste0("beta_", f), paste0("β for ", f), value = val, step = 0.1)
    })
  })
  
  observeEvent(input$reset_betas, {
    fnames <- feature_levels(input$p_features)
    keep <- intersect(names(DEFAULT_BETAS), fnames)
    updateSelectizeInput(session, "relevant_features", choices = fnames, selected = keep)
    for (f in keep) updateNumericInput(session, paste0("beta_", f), value = unname(DEFAULT_BETAS[f]))
  })
  
  beta_true_vec <- reactive({
    p <- input$p_features; req(p)
    b <- rep(0, p); names(b) <- feature_levels(p)
    for (f in intersect(input$relevant_features, names(b))) {
      v <- input[[paste0("beta_", f)]]
      if (is.null(v) || is.na(v)) v <- if (f %in% names(DEFAULT_BETAS)) unname(DEFAULT_BETAS[f]) else 1
      b[[f]] <- v
    }
    b
  })
  
  output$beta_preview <- renderUI({
    b <- beta_true_vec(); nz <- b[b != 0]
    txt <- if (length(nz) == 0) "all β = 0" else
      paste(paste0(names(nz), " = ", round(nz, 3)), collapse = ",  ")
    HTML(paste0("<div class='beta-preview'>", txt, "</div>"))
  })
  
  output$signal_summary <- renderUI({
    b <- beta_true_vec(); nz <- b[b != 0]
    HTML(paste0("<h4 class='number' style='font-size: 1.4em; color:#06d6a0;'>",
                length(nz), " of ", length(b), " relevant</h4>",
                "<div style='color:#8892b0; font-size:0.85em;'>",
                if (length(nz) == 0) "no signal" else paste(names(nz), collapse = ", "),
                "</div>"))
  })
  
  # ---------- Simulation ----------
  observeEvent(input$run_pipeline, {
    if (results$running) return()
    results$running <- TRUE
    updateActionButton(session, "run_pipeline", label = "Processing...", icon = icon("spinner"))
    updateProgressBar(session = session, id = "pipeline_progress", value = 0, title = "Initializing simulation execution...")
    set.seed(input$seed_val); R <- input$r_runs; folds <- input$nfolds; meta <- current_meta()
    results$feature_names <- meta$features; results$p_used <- meta$p
    
    lasso_coef_mat  <- matrix(0, nrow = R, ncol = meta$p); lasso_selec_mat <- matrix(0, nrow = R, ncol = meta$p)
    ridge_coef_mat  <- matrix(0, nrow = R, ncol = meta$p); ridge_selec_mat <- matrix(0, nrow = R, ncol = meta$p)
    
    beta_true <- unname(beta_true_vec()); results$beta_true <- beta_true
    signal_txt <- { b <- beta_true_vec(); nz <- b[b != 0]
    if (length(nz) == 0) "none (pure noise)" else paste(paste0(names(nz), "=", round(nz, 3)), collapse = ", ") }
    
    results$log <- c("Constructing simulation framework execution layers...",
                     paste("Engine Configuration Mode =", meta$mode_label),
                     paste("Observations n =", meta$n), paste("Features p =", meta$p),
                     paste("Replications R =", R), paste("Cross-Validation folds =", folds),
                     paste("Multicollinearity γ =", input$gamma_mc),
                     paste("Relevant features (true β ≠ 0):", signal_txt))
    
    withProgress(message = 'Fitting Regularized Systems...', value = 0, {
      for (i in 1:R) {
        X_dum <- matrix(rnorm(meta$n * (meta$p + 1)), nrow = meta$n, ncol = meta$p + 1)
        Z <- X_dum[, 1:meta$p]; z_p1 <- X_dum[, meta$p + 1]
        X_active <- sqrt(1 - input$gamma_mc^2) * Z + input$gamma_mc * z_p1
        y_active <- X_active %*% beta_true + rnorm(meta$n)
        colnames(X_active) <- feature_levels(meta$p)
        if (i == 1) { results$X_data <- X_active; results$y_data <- y_active }
        
        cv_l <- tryCatch(cv.glmnet(X_active, y_active, alpha = 1, nfolds = folds, family = "gaussian", intercept = TRUE), error = function(e) NULL)
        if (!is.null(cv_l)) { mod_l <- glmnet(X_active, y_active, alpha = 1, lambda = cv_l$lambda.min, family = "gaussian", intercept = TRUE)
        l_coef <- as.numeric(coef(mod_l)[-1]) } else l_coef <- rep(0, meta$p)
        
        cv_r <- tryCatch(cv.glmnet(X_active, y_active, alpha = 0, nfolds = folds, family = "gaussian", intercept = TRUE), error = function(e) NULL)
        if (!is.null(cv_r)) { mod_r <- glmnet(X_active, y_active, alpha = 0, lambda = cv_r$lambda.min, family = "gaussian", intercept = TRUE)
        r_coef <- as.numeric(coef(mod_r)[-1]) } else r_coef <- rep(0, meta$p)
        
        lasso_coef_mat[i, ] <- l_coef; lasso_selec_mat[i, ] <- as.numeric(abs(l_coef) > 1e-5)
        ridge_coef_mat[i, ] <- r_coef; ridge_selec_mat[i, ] <- as.numeric(abs(r_coef) > 1e-5)
        if (i %% max(1, floor(R/10)) == 0) updateProgressBar(session = session, id = "pipeline_progress", value = round((i/R)*100))
        incProgress(1/R)
      }
    })
    
    process_profile_metrics <- function(coefs, selec, R) {
      means <- colMeans(coefs); vars <- apply(coefs, 2, var); probs <- colMeans(selec)
      sample_limit <- min(R, 100); jaccard_vals <- c()
      for (m in 1:(sample_limit - 1)) for (k in (m + 1):sample_limit) {
        set_m <- which(selec[m, ] == 1); set_k <- which(selec[k, ] == 1)
        jaccard_vals <- c(jaccard_vals,
                          if (length(union(set_m, set_k)) > 0) length(intersect(set_m, set_k)) / length(union(set_m, set_k)) else 1)
      }
      list(means = means, vars = vars, probs = probs, jaccard = mean(jaccard_vals),
           coef_mat = coefs, selec_mat = selec)
    }
    
    results$lasso <- process_profile_metrics(lasso_coef_mat, lasso_selec_mat, R)
    results$ridge <- process_profile_metrics(ridge_coef_mat, ridge_selec_mat, R)
    results$cor_matrix <- cor(results$X_data)
    
    results$log <- c(results$log, "✓ Processing calculations finalized.",
                     paste("LASSO Mean Stability index:", round(results$lasso$jaccard, 4)),
                     paste("Ridge Mean Stability index:", round(results$ridge$jaccard, 4)),
                     "✓ Complete Profile Generated.")
    results$running <- FALSE
    updateActionButton(session, "run_pipeline", label = "Execute Simulation", icon = icon("play"))
    updateProgressBar(session = session, id = "pipeline_progress", value = 100, title = "Completed!")
  })
  
  observeEvent(session, { click("run_pipeline") }, once = TRUE)
  
  # ---------- Value boxes (REMOVED) ----------
  # The four dashboard value boxes (Replications / Relevant-Total Features /
  # Observations / Framework Status) have been removed as requested.
  # Note: shinydashboard::valueBoxOutput and renderValueBox definitions
  # are no longer used in the UI for the Pipeline Dashboard tab.
  
  output$pipeline_log <- renderPrint({ cat(paste(results$log, collapse = "\n")) })
  
  output$engine_summary_table <- renderDT({
    meta <- current_meta(); b <- beta_true_vec(); nz <- b[b != 0]
    df <- data.frame(
      Parameter = c("Engine Mode", "Observations (n)", "Features (p)",
                    "Multicollinearity (γ)", "Replications (R)", "CV Folds",
                    "Stability Threshold", "Relevant Features", "True β (non-zero)"),
      Value = c(meta$mode_label, meta$n, meta$p, input$gamma_mc, input$r_runs,
                input$nfolds, input$pi_threshold,
                if (length(nz) == 0) "none" else paste(names(nz), collapse = ", "),
                if (length(nz) == 0) "-" else paste(round(nz, 3), collapse = ", ")))
    datatable(df, options = list(dom = 't', pageLength = 9), rownames = FALSE)
  })
  
  output$true_beta_plot <- renderPlotly({
    b <- beta_true_vec(); p_now <- length(b)
    df <- data.frame(Feature = order_features(names(b), p_now),
                     Beta = as.numeric(b),
                     Role = ifelse(b != 0, "Relevant", "Noise"))
    g <- ggplot(df, aes(x = Feature, y = Beta, fill = Role)) +
      geom_bar(stat = "identity", alpha = 0.9, width = 0.6) +
      geom_hline(yintercept = 0, color = "#e1e8f0", linewidth = 0.4) +
      scale_fill_manual(values = c("Relevant" = "#06d6a0", "Noise" = "#2a3a5c")) +
      theme_minimal() +
      theme(plot.background = element_rect(fill = "transparent", color = NA),
            panel.background = element_rect(fill = "transparent", color = NA),
            axis.text.x = element_text(angle = 45, hjust = 1, color = "#e1e8f0"),
            axis.text.y = element_text(color = "#e1e8f0"),
            text = element_text(color = "#e1e8f0"),
            panel.grid.major = element_line(color = "#2a3a5c", linewidth = 0.2)) +
      labs(title = "Specified True Coefficients (β)", x = "", y = "True β")
    ggplotly(g) %>% layout(plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)")
  })
  
  # ---------- Standard plots ----------
  render_selection_plot <- function(stats_obj, title_text) {
    req(stats_obj); m <- run_meta()
    df <- data.frame(Feature = order_features(m$features, m$p), Probability = stats_obj$probs,
                     Classification = ifelse(stats_obj$probs >= input$pi_threshold, "Stable", "Unstable"))
    p <- ggplot(df, aes(x = Feature, y = Probability, fill = Classification)) +
      geom_bar(stat = "identity", alpha = 0.85, width = 0.6) +
      geom_hline(yintercept = input$pi_threshold, color = "#ef476f", linetype = "dashed", linewidth = 0.8) +
      scale_fill_manual(values = c("Stable" = "#06d6a0", "Unstable" = "#118ab2")) +
      scale_x_discrete(drop = FALSE) + theme_minimal() +
      theme(plot.background = element_rect(fill = "transparent", color = NA),
            panel.background = element_rect(fill = "transparent", color = NA),
            axis.text.x = element_text(angle = 45, hjust = 1, color = "#e1e8f0"),
            axis.text.y = element_text(color = "#e1e8f0"),
            text = element_text(color = "#e1e8f0"),
            panel.grid.major = element_line(color = "#2a3a5c", linewidth = 0.2)) +
      labs(title = title_text, y = "Selection Probability", x = "") + ylim(0, 1)
    ggplotly(p) %>% layout(plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)")
  }
  output$lasso_selection_prob <- renderPlotly({ render_selection_plot(results$lasso, "LASSO Feature Selection Probability Profiler") })
  output$ridge_selection_prob <- renderPlotly({ render_selection_plot(results$ridge, "Ridge Feature Selection Probability Profiler") })
  
  render_convergence_trace <- function(stats_obj, title_text) {
    req(stats_obj); m <- run_meta()
    cum_mean <- apply(stats_obj$selec_mat, 2, cumsum) / (1:nrow(stats_obj$selec_mat))
    colnames(cum_mean) <- m$features
    df_long <- as.data.frame(cum_mean) %>% mutate(Iteration = 1:n()) %>%
      pivot_longer(cols = -Iteration, names_to = "Feature", values_to = "Probability") %>%
      mutate(Feature = order_features(Feature, m$p)) %>% arrange(Feature, Iteration)
    p <- ggplot(df_long, aes(x = Iteration, y = Probability, color = Feature)) +
      geom_line(linewidth = 0.65, alpha = 0.8) + scale_color_viridis_d(drop = FALSE) +
      theme_minimal() +
      theme(plot.background = element_rect(fill = "transparent", color = NA),
            panel.background = element_rect(fill = "transparent", color = NA),
            text = element_text(color = "#e1e8f0"),
            panel.grid.major = element_line(color = "#2a3a5c", linewidth = 0.2)) +
      labs(title = title_text, x = "Iteration Progression", y = "Cumulative Selection Probability")
    ggplotly(p) %>% layout(plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)")
  }
  output$lasso_convergence <- renderPlotly({ render_convergence_trace(results$lasso, "LASSO Convergence Track") })
  output$ridge_convergence <- renderPlotly({ render_convergence_trace(results$ridge, "Ridge Convergence Track") })
  
  render_distribution_boxplots <- function(stats_obj, title_text) {
    req(stats_obj); m <- run_meta()
    df_coefs <- as.data.frame(stats_obj$coef_mat); colnames(df_coefs) <- m$features
    df_long <- df_coefs %>% pivot_longer(cols = everything(), names_to = "Feature", values_to = "Estimate") %>%
      mutate(Feature = order_features(Feature, m$p))
    df_true <- data.frame(Feature = order_features(m$features, m$p), Estimate = as.numeric(results$beta_true))
    p <- ggplot(df_long, aes(x = Feature, y = Estimate, fill = Feature)) +
      geom_boxplot(outlier.size = 0.6, alpha = 0.7, color = "#e1e8f0") +
      geom_point(data = df_true, aes(x = Feature, y = Estimate), inherit.aes = FALSE,
                 color = "#ef476f", size = 2, shape = 18) +
      scale_x_discrete(drop = FALSE) + theme_minimal() +
      theme(plot.background = element_rect(fill = "transparent", color = NA),
            panel.background = element_rect(fill = "transparent", color = NA),
            axis.text.x = element_text(angle = 45, hjust = 1, color = "#e1e8f0"),
            text = element_text(color = "#e1e8f0"),
            legend.position = "none",
            panel.grid.major = element_line(color = "#2a3a5c", linewidth = 0.2)) +
      labs(title = paste0(title_text, " (red diamond = true β)"), x = "", y = "Coefficient Estimates")
    ggplotly(p) %>% layout(plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)")
  }
  output$lasso_coefficients <- renderPlotly({ render_distribution_boxplots(results$lasso, "LASSO Estimate Configurations") })
  output$ridge_coefficients <- renderPlotly({ render_distribution_boxplots(results$ridge, "Ridge Estimate Configurations") })
  
  render_summary_table <- function(stats_obj) {
    req(stats_obj); m <- run_meta()
    df <- data.frame(Feature = m$features,
                     True_Beta = as.numeric(results$beta_true),
                     Mean_Beta = stats_obj$means,
                     Bias = stats_obj$means - as.numeric(results$beta_true),
                     Variance_Beta = stats_obj$vars,
                     Selection_Probability = stats_obj$probs)
    datatable(df, options = list(pageLength = 10, dom = 'tip', ordering = FALSE), rownames = FALSE) %>%
      formatRound(columns = c("True_Beta", "Mean_Beta", "Bias", "Variance_Beta", "Selection_Probability"), digits = 4)
  }
  output$lasso_coef_table <- renderDT({ render_summary_table(results$lasso) })
  output$ridge_coef_table <- renderDT({ render_summary_table(results$ridge) })
  
  output$lasso_jaccard <- renderUI({ req(results$lasso); HTML(paste0("<div class='number stability-high'>", round(results$lasso$jaccard, 4), "</div>")) })
  output$ridge_jaccard <- renderUI({ req(results$ridge); HTML(paste0("<div class='number' style='color:#ffd166;'>", round(results$ridge$jaccard, 4), "</div>")) })
  output$lasso_important <- renderUI({ req(results$lasso); m <- run_meta()
  f <- m$features[which(results$lasso$probs >= input$pi_threshold)]
  HTML(paste0("<div style='font-size:1.1em; font-weight:bold; color:#06d6a0;'>",
              if (length(f) > 0) paste(sort_features(f), collapse = ", ") else "None Matched", "</div>")) })
  output$ridge_important <- renderUI({ req(results$ridge); m <- run_meta()
  f <- m$features[which(results$ridge$probs >= input$pi_threshold)]
  HTML(paste0("<div style='font-size:1.1em; font-weight:bold; color:#ffd166;'>",
              if (length(f) > 0) paste(sort_features(f), collapse = ", ") else "None Matched", "</div>")) })
  output$lasso_probs <- renderPrint({ req(results$lasso); pr <- round(results$lasso$probs, 4); names(pr) <- run_meta()$features; print(pr) })
  output$ridge_probs <- renderPrint({ req(results$ridge); pr <- round(results$ridge$probs, 4); names(pr) <- run_meta()$features; print(pr) })
  
  output$compare_selection <- renderPlotly({
    req(results$lasso, results$ridge); m <- run_meta()
    df <- data.frame(Feature = order_features(rep(m$features, 2), m$p),
                     Probability = c(results$lasso$probs, results$ridge$probs),
                     Method = rep(c("LASSO", "Ridge"), each = m$p))
    p <- ggplot(df, aes(x = Feature, y = Probability, fill = Method)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.85) +
      scale_fill_manual(values = c("LASSO" = "#00b4d8", "Ridge" = "#ffd166")) +
      scale_x_discrete(drop = FALSE) + theme_minimal() +
      labs(title = "Selection Probabilities Comparison", y = "Probability", x = "") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "#e1e8f0"),
            text = element_text(color = "#e1e8f0"),
            panel.grid.major = element_line(color = "#2a3a5c", linewidth = 0.2))
    ggplotly(p) %>% layout(plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)")
  })
  output$compare_jaccard <- renderPlotly({
    req(results$lasso, results$ridge)
    df <- data.frame(Method = c("LASSO", "Ridge"), Stability = c(results$lasso$jaccard, results$ridge$jaccard))
    p <- ggplot(df, aes(x = Method, y = Stability, fill = Method)) +
      geom_bar(stat = "identity", width = 0.4, alpha = 0.9) +
      scale_fill_manual(values = c("LASSO" = "#00b4d8", "Ridge" = "#ffd166")) +
      theme_minimal() + labs(title = "Jaccard Index Comparison", y = "Stability Index", x = "") +
      theme(text = element_text(color = "#e1e8f0"),
            panel.grid.major = element_line(color = "#2a3a5c", linewidth = 0.2)) + ylim(0, 1)
    ggplotly(p) %>% layout(plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)")
  })
  output$compare_bias <- renderPlotly({
    req(results$lasso, results$ridge, results$beta_true); m <- run_meta()
    df <- data.frame(Feature = order_features(rep(m$features, 2), m$p),
                     Bias = c(results$lasso$means - results$beta_true, results$ridge$means - results$beta_true),
                     Method = rep(c("LASSO", "Ridge"), each = m$p))
    p <- ggplot(df, aes(x = Feature, y = Bias, fill = Method)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.85) +
      geom_hline(yintercept = 0, color = "#e1e8f0", linewidth = 0.5) +
      scale_fill_manual(values = c("LASSO" = "#00b4d8", "Ridge" = "#ffd166")) +
      scale_x_discrete(drop = FALSE) + theme_minimal() +
      labs(title = "Coefficient Bias Comparison (True β Known)", y = "Bias", x = "") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, color = "#e1e8f0"),
            text = element_text(color = "#e1e8f0"),
            panel.grid.major = element_line(color = "#2a3a5c", linewidth = 0.2))
    ggplotly(p) %>% layout(plot_bgcolor = "rgba(0,0,0,0)", paper_bgcolor = "rgba(0,0,0,0)")
  })
  output$comparison_table <- renderDT({
    req(results$lasso, results$ridge)
    beta <- as.numeric(results$beta_true); true_set <- which(beta != 0)
    stable_set <- function(probs) which(probs >= input$pi_threshold)
    tpr <- function(probs) if (length(true_set) == 0) NA else length(intersect(stable_set(probs), true_set)) / length(true_set)
    fpr <- function(probs) { noise_set <- setdiff(seq_along(beta), true_set)
    if (length(noise_set) == 0) NA else length(intersect(stable_set(probs), noise_set)) / length(noise_set) }
    df <- data.frame(Model_Method = c("LASSO (L1 Penalty)", "Ridge (L2 Penalty)"),
                     Mean_Jaccard = c(results$lasso$jaccard, results$ridge$jaccard),
                     Stable_Features = c(length(stable_set(results$lasso$probs)), length(stable_set(results$ridge$probs))),
                     True_Positive_Rate = c(tpr(results$lasso$probs), tpr(results$ridge$probs)),
                     False_Positive_Rate = c(fpr(results$lasso$probs), fpr(results$ridge$probs)))
    datatable(df, options = list(dom = 't'), rownames = FALSE) %>%
      formatRound(columns = c("Mean_Jaccard", "True_Positive_Rate", "False_Positive_Rate"), digits = 4)
  })
  
  # ============================================================
  # 2D GEOMETRIC VIEW
  # ============================================================
  
  ridge_solution_2d <- function(b_ols, t) {
    nrm <- sqrt(sum(b_ols^2))
    if (nrm <= t || nrm < 1e-12) return(b_ols)
    b_ols * (t / nrm)
  }
  
  lasso_solution_2d <- function(b_ols, t) {
    b1 <- b_ols[1]; b2 <- b_ols[2]
    if (abs(b1) + abs(b2) <= t + 1e-12) return(b_ols)
    cand <- list(c(b1 + (t - b1 - b2)/2, b2 + (t - b1 - b2)/2),
                 c(b1 - (t + b1 - b2)/2, b2 + (t + b1 - b2)/2),
                 c(b1 + (t - b1 + b2)/2, b2 - (t - b1 + b2)/2),
                 c(b1 - (t + b1 + b2)/2, b2 - (t + b1 + b2)/2))
    feasible <- Filter(function(p) abs(p[1]) + abs(p[2]) <= t + 1e-6, cand)
    if (length(feasible) == 0) {
      if (abs(b1) >= abs(b2)) return(c(sign(b1) * t, 0))
      return(c(0, sign(b2) * t))
    }
    dists <- sapply(feasible, function(p) sum((p - b_ols)^2))
    feasible[[which.min(dists)]]
  }
  
  rss_ellipse <- function(cx, cy, r, rho, n = 220) {
    Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
    L <- chol(Sigma)
    theta <- seq(0, 2*pi, length.out = n)
    u <- rbind(cos(theta), sin(theta))
    pts <- t(L) %*% u * r
    data.frame(x = cx + pts[1, ], y = cy + pts[2, ])
  }
  
  rss_radius <- function(b_ols, sol, rho) {
    Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
    d <- sol - b_ols
    sqrt(as.numeric(t(d) %*% solve(Sigma) %*% d))
  }
  
  build_geom_panel <- function(type, b1_ols, b2_ols, t_val, rho,
                               show_ellipses, show_guides, show_path, show_corner) {
    b_ols <- c(b1_ols, b2_ols)
    sol <- if (type == "ridge") ridge_solution_2d(b_ols, t_val) else
      lasso_solution_2d(b_ols, t_val)
    
    r_tang <- rss_radius(b_ols, sol, rho)
    
    lim <- max(abs(c(b1_ols, b2_ols, t_val))) * 1.5 + 0.8
    lim <- max(lim, 3)
    
    p <- plot_ly()
    
    if (show_ellipses) {
      radii <- seq(r_tang * 0.25, lim * 1.1, length.out = 7)
      for (i in seq_along(radii)) {
        ell <- rss_ellipse(b1_ols, b2_ols, radii[i], rho)
        is_tangent <- abs(radii[i] - r_tang) < (lim * 0.06)
        p <- p %>% add_trace(
          x = ell$x, y = ell$y, type = "scatter", mode = "lines",
          line = list(
            color = if (is_tangent) "#ef476f" else "rgba(180,200,220,0.5)",
            width = if (is_tangent) 3 else 1,
            dash  = if (is_tangent) "solid" else "dot"
          ),
          name = if (is_tangent) paste0("RSS contour at tangency (level ~", round(radii[i], 2), ")")
          else paste0("RSS level ~", round(radii[i], 2)),
          showlegend = is_tangent,
          hoverinfo = "name"
        )
      }
    }
    
    if (type == "ridge") {
      theta <- seq(0, 2*pi, length.out = 320)
      p <- p %>% add_trace(
        x = t_val * cos(theta), y = t_val * sin(theta),
        type = "scatter", mode = "lines", fill = "toself",
        fillcolor = "rgba(255, 209, 102, 0.15)",
        line = list(color = "#ffd166", width = 3),
        name = paste0("Ridge constraint  b1^2+b2^2 <= ", t_val^2),
        hoverinfo = "name"
      )
    } else {
      p <- p %>% add_trace(
        x = c(t_val, 0, -t_val, 0, t_val), y = c(0, t_val, 0, -t_val, 0),
        type = "scatter", mode = "lines", fill = "toself",
        fillcolor = "rgba(0, 180, 216, 0.15)",
        line = list(color = "#00b4d8", width = 3),
        name = paste0("LASSO constraint  |b1|+|b2| <= ", t_val),
        hoverinfo = "name"
      )
    }
    
    p <- p %>% add_trace(
      x = b1_ols, y = b2_ols, type = "scatter", mode = "markers",
      marker = list(size = 14, color = "#ef476f", symbol = "x",
                    line = list(color = "#fff", width = 1.5)),
      name = paste0("OLS bhat  (", round(b1_ols, 2), ", ", round(b2_ols, 2), ")"),
      hoverinfo = "name"
    )
    
    is_sparse_lasso <- type == "lasso" && (abs(sol[1]) < 1e-6 || abs(sol[2]) < 1e-6)
    p <- p %>% add_trace(
      x = sol[1], y = sol[2], type = "scatter", mode = "markers",
      marker = list(size = 16,
                    color = if (type == "ridge") "#ffd166" else "#00b4d8",
                    symbol = if (is_sparse_lasso) "diamond" else "circle",
                    line = list(color = "#fff", width = 2)),
      name = paste0(if (type == "ridge") "Ridge" else "LASSO", " solution (",
                    round(sol[1], 3), ", ", round(sol[2], 3), ")",
                    if (is_sparse_lasso) "  <- SPARSE" else ""),
      hoverinfo = "name"
    )
    
    if (show_guides) {
      p <- p %>% add_trace(
        x = c(0, sol[1]), y = c(sol[2], sol[2]),
        type = "scatter", mode = "lines",
        line = list(color = "rgba(255,255,255,0.45)", width = 1, dash = "dot"),
        name = "b2 guide", showlegend = FALSE, hoverinfo = "skip"
      )
      p <- p %>% add_trace(
        x = c(sol[1], sol[1]), y = c(0, sol[2]),
        type = "scatter", mode = "lines",
        line = list(color = "rgba(255,255,255,0.45)", width = 1, dash = "dot"),
        name = "b1 guide", showlegend = FALSE, hoverinfo = "skip"
      )
    }
    
    if (show_path) {
      p <- p %>% add_trace(
        x = c(b1_ols, sol[1]), y = c(b2_ols, sol[2]),
        type = "scatter", mode = "lines",
        line = list(color = "#ef476f", width = 1.5, dash = "dash"),
        name = "Shrinkage path", showlegend = FALSE, hoverinfo = "skip"
      )
    }
    
    if (type == "lasso" && show_corner && is_sparse_lasso) {
      corner <- c(0, 0)
      if (abs(sol[1]) > 1e-6) corner <- c(sign(sol[1]) * t_val, 0)
      if (abs(sol[2]) > 1e-6) corner <- c(0, sign(sol[2]) * t_val)
      p <- p %>% add_trace(
        x = corner[1], y = corner[2], type = "scatter", mode = "markers",
        marker = list(size = 26, color = "rgba(239, 71, 111, 0.0)",
                      line = list(color = "#ef476f", width = 3)),
        name = "Diamond corner hit (feature set to 0)",
        hoverinfo = "name"
      )
    }
    
    title_txt <- if (type == "ridge")
      "<b style='color:#ffd166;'>Ridge (L2)</b> - ellipse touches the <b>smooth circle</b>: tangency is off-axis -> shrinkage only"
    else
      "<b style='color:#00b4d8;'>LASSO (L1)</b> - ellipse reaches the <b>diamond</b>: tangency often lands on a corner (axis) -> feature selection"
    
    p %>% layout(
      title = list(text = title_txt, font = list(color = "#e1e8f0", size = 12)),
      xaxis = list(title = "b1", color = "#e1e8f0", gridcolor = "#2a3a5c",
                   zerolinecolor = "#4a5568", zerolinewidth = 2,
                   range = c(-lim, lim), constrain = "domain"),
      yaxis = list(title = "b2", color = "#e1e8f0", gridcolor = "#2a3a5c",
                   zerolinecolor = "#4a5568", zerolinewidth = 2,
                   range = c(-lim, lim), scaleanchor = "x", constrain = "domain"),
      paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(11, 15, 26, 0.6)",
      font = list(color = "#e1e8f0"),
      legend = list(x = 0.02, y = 0.98, bgcolor = "rgba(26, 26, 46, 0.85)",
                    bordercolor = "#2a3a5c", borderwidth = 1,
                    font = list(color = "#e1e8f0", size = 10)),
      margin = list(l = 40, r = 20, b = 40, t = 60)
    )
  }
  
  output$geom_2d_container <- renderUI({
    mode <- input$geom_display
    if (mode == "Ridge & LASSO (side by side)") {
      fluidRow(
        column(6, plotlyOutput("geom_2d_ridge", height = "520px")),
        column(6, plotlyOutput("geom_2d_lasso", height = "520px"))
      )
    } else if (mode == "Ridge only") {
      plotlyOutput("geom_2d_ridge", height = "540px")
    } else {
      plotlyOutput("geom_2d_lasso", height = "540px")
    }
  })
  
  output$geom_2d_ridge <- renderPlotly({
    build_geom_panel("ridge",
                     input$geom_b1_ols, input$geom_b2_ols, input$geom_t, input$geom_rho,
                     input$geom_show_ellipses, input$geom_show_guides,
                     input$geom_show_path, input$geom_show_corner)
  })
  
  output$geom_2d_lasso <- renderPlotly({
    build_geom_panel("lasso",
                     input$geom_b1_ols, input$geom_b2_ols, input$geom_t, input$geom_rho,
                     input$geom_show_ellipses, input$geom_show_guides,
                     input$geom_show_path, input$geom_show_corner)
  })
  
  output$geom_ridge_solution <- renderUI({
    sol <- ridge_solution_2d(c(input$geom_b1_ols, input$geom_b2_ols), input$geom_t)
    HTML(paste0("( ", round(sol[1], 4), " , ", round(sol[2], 4), " )"))
  })
  output$geom_lasso_solution <- renderUI({
    sol <- lasso_solution_2d(c(input$geom_b1_ols, input$geom_b2_ols), input$geom_t)
    flag <- if (abs(sol[1]) < 1e-6 || abs(sol[2]) < 1e-6)
      " <span class='sparse-flag'>[SPARSE]</span>" else
        " <span class='dense-flag'>[dense]</span>"
    HTML(paste0("( ", round(sol[1], 4), " , ", round(sol[2], 4), " )", flag))
  })
  
  output$geom_ridge_tangency <- renderUI({
    sol <- ridge_solution_2d(c(input$geom_b1_ols, input$geom_b2_ols), input$geom_t)
    on_axis <- abs(sol[1]) < 1e-3 || abs(sol[2]) < 1e-3
    txt <- paste0("Tangency at (", round(sol[1], 4), ", ", round(sol[2], 4), ") - ",
                  if (on_axis) "on axis (rare for Ridge)"
                  else paste0("off-axis, distance to nearest axis ~ ",
                              round(min(abs(sol[1]), abs(sol[2])), 4)))
    HTML(paste0("<div style='color:#ffd166;'>", txt, "</div>"))
  })
  
  output$geom_lasso_tangency <- renderUI({
    sol <- lasso_solution_2d(c(input$geom_b1_ols, input$geom_b2_ols), input$geom_t)
    on_axis <- abs(sol[1]) < 1e-6 || abs(sol[2]) < 1e-6
    if (on_axis) {
      txt <- paste0("Tangency at corner (", round(sol[1], 4), ", ", round(sol[2], 4),
                    ") - lands exactly on the b",
                    if (abs(sol[1]) < 1e-6) "2" else "1", " axis -> coefficient set to 0")
    } else {
      txt <- paste0("Tangency at (", round(sol[1], 4), ", ", round(sol[2], 4),
                    ") - on a face, not a corner (no exact zero). ",
                    "Increase OLS magnitude or decrease t to hit a corner.")
    }
    HTML(paste0("<div style='color:#00b4d8;'>", txt, "</div>"))
  })
  
  observeEvent(input$geom_auto_sparse, {
    t_val <- input$geom_t
    b1 <- t_val * 1.6
    b2 <- t_val * 0.35
    updateNumericInput(session, "geom_b1_ols", value = round(b1, 2))
    updateNumericInput(session, "geom_b2_ols", value = round(b2, 2))
  })
  
  # ============================================================
  # Multicollinearity
  # ============================================================
  output$correlation_plot <- renderPlot({
    req(results$cor_matrix)
    corrplot(results$cor_matrix, method = "color", type = "upper", order = "original",
             addCoef.col = "#e1e8f0", tl.col = "#00b4d8", number.cex = 0.85,
             bg = "#0b0f1a", cl.pos = "b")
  })
  output$obs_display      <- renderUI({ HTML(paste0("<div class='number' style='font-size:1.2em;'>", current_meta()$n, "</div>")) })
  output$features_display <- renderUI({ HTML(paste0("<div class='number' style='font-size:1.2em;'>", current_meta()$p, "</div>")) })
  output$mean_corr <- renderUI({ req(results$cor_matrix); c_mat <- results$cor_matrix
  HTML(paste0("<div class='number' style='font-size:1.2em; color:#06d6a0;'>",
              round(mean(abs(c_mat[lower.tri(c_mat)])), 4), "</div>")) })
  output$condition_num <- renderUI({ req(results$cor_matrix); ev <- eigen(results$cor_matrix)$values
  HTML(paste0("<div class='number' style='font-size:1.2em; color:#ef476f;'>",
              round(sqrt(max(ev) / max(min(ev), 1e-8)), 4), "</div>")) })
  
  # ============================================================
  # Data Matrix Explorer
  # ============================================================
  output$data_matrix_table <- renderDT({
    req(results$lasso, results$ridge)
    type <- input$matrix_type; rows_to_show <- min(input$show_rows, 100)
    if (type == "Design Matrix (X)") {
      req(results$X_data); df <- as.data.frame(head(results$X_data, rows_to_show))
      datatable(df, options = list(pageLength = 10, scrollX = TRUE, dom = 'rtip'), class = "cell-border stripe")
    } else if (type == "Correlation Matrix") {
      req(results$cor_matrix); df <- as.data.frame(round(results$cor_matrix, 4))
      datatable(df, options = list(pageLength = 10, scrollX = TRUE, dom = 'rt'), class = "cell-border stripe") %>%
        formatStyle(columns = names(df),
                    backgroundColor = styleInterval(c(-0.5, 0.5),
                                                    c('rgba(239, 71, 111, 0.15)', 'rgba(26, 26, 46, 0.8)', 'rgba(6, 214, 160, 0.15)')))
    } else if (type == "Selection Matrix") {
      req(results$lasso$selec_mat)
      df_selec <- as.data.frame(head(results$lasso$selec_mat, rows_to_show))
      colnames(df_selec) <- results$feature_names
      df_display <- cbind(Iteration = 1:nrow(df_selec), df_selec)
      datatable(df_display, rownames = FALSE,
                caption = htmltools::tags$caption(
                  style = 'caption-side: top; color: #ffd166; font-weight: bold; margin-bottom: 5px;',
                  'LASSO Feature Selection Matrix (1 = Selected, 0 = Omitted)'),
                options = list(pageLength = 10, scrollX = TRUE, dom = 'rtip',
                               columnDefs = list(list(className = 'dt-center', targets = "_all")))) %>%
        formatStyle(columns = results$feature_names,
                    backgroundColor = styleEqual(c(0, 1), c('#161c2e', '#06d6a0')),
                    color = styleEqual(c(0, 1), c('#4e5d78', '#0b0f1a')),
                    fontWeight = styleEqual(c(0, 1), c('normal', 'bold')),
                    border = '1px solid #2a3a5c') %>%
        formatStyle('Iteration', backgroundColor = '#1a2a4a', color = '#00b4d8', fontWeight = 'bold')
    } else if (type == "True Coefficients (β)") {
      req(results$beta_true); m <- run_meta()
      df <- data.frame(Feature = m$features,
                       True_Beta = as.numeric(results$beta_true),
                       Role = ifelse(as.numeric(results$beta_true) != 0, "Relevant", "Noise"))
      datatable(df, rownames = FALSE, options = list(pageLength = 10, dom = 'rtip', ordering = FALSE),
                class = "cell-border stripe") %>%
        formatRound(columns = "True_Beta", digits = 4) %>%
        formatStyle("Role",
                    backgroundColor = styleEqual(c("Relevant", "Noise"), c('rgba(6, 214, 160, 0.25)', '#161c2e')),
                    fontWeight = styleEqual(c("Relevant", "Noise"), c('bold', 'normal')))
    }
  })
  
  output$matrix_dims <- renderText({
    req(results$lasso); type <- input$matrix_type
    if (type == "Design Matrix (X)") { req(results$X_data)
      paste0("Dimensions: ", nrow(results$X_data), " rows x ", ncol(results$X_data), " variables")
    } else if (type == "Correlation Matrix") { req(results$cor_matrix)
      paste0("Dimensions: ", nrow(results$cor_matrix), " x ", ncol(results$cor_matrix), " symmetric matrix")
    } else if (type == "Selection Matrix") { req(results$lasso$selec_mat)
      paste0("Dimensions: ", nrow(results$lasso$selec_mat), " simulation iterations (R) x ",
             ncol(results$lasso$selec_mat), " candidate features (p)")
    } else if (type == "True Coefficients (β)") { req(results$beta_true)
      paste0("Length: ", length(results$beta_true), " coefficients (",
             sum(results$beta_true != 0), " non-zero / relevant)")
    }
  })
}

shinyApp(ui = ui, server = server)