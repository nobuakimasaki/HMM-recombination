library(tidyverse)
library(stringr)
library(readr)
library(purrr)
library(data.table)

### Preliminary analysis ### 

# Result of applying HMM to the data (detected recombinant + non-recombinant sequences)
res <- read.csv("../summary_files/combined_optimization_results_with_summary.csv") %>%
  mutate(across(c(test_start, test_end), as.Date)) %>%
  mutate(date = test_start + (test_end - test_start) / 2)

# Define new dataframe with parental lineage count
res_w_n_parents <- res %>%
  mutate(
    n_parents = case_when(
      is.na(parental_lineages) | parental_lineages == "" ~ 0L,
      TRUE ~ str_count(parental_lineages, fixed(";")) + 1L
      )
    )

# Keep exactly 2 parental lineages
res_two_parents <- res_w_n_parents %>%
  filter(n_parents == 2) %>%
  select(-n_parents)

# Count number of parents
tally_parents <- res_w_n_parents %>%
  count(n_parents, name = "n_rows") %>%
  arrange(n_parents)

# Compute numerator and denominator
num <- nrow(res_two_parents)
den <- sum(tally_parents$n_rows[-1])

# Exact (Clopper–Pearson) binomial CI
bt <- binom.test(num, den)

bt$estimate   # estimated proportion
bt$conf.int   # exact 95% Clopper–Pearson CI

# We also simply want the proportion of recombinants across all of our samples
num2 <- sum(tally_parents$n_rows[tally_parents$n_parents != 1])
den2 <- nrow(res)

# Exact (Clopper–Pearson) binomial CI
bt2 <- binom.test(num2, den2)

bt2$estimate   # estimated proportion
bt2$conf.int   # exact 95% Clopper–Pearson CI

### Plotting the detected recombinant proportion against ONS prevalence ### 

# Filtering to cases with two or more parents
res_two_or_more_parents <- res_w_n_parents %>%
  add_count(date, name = "n_samples") %>%         
  filter(n_parents >= 2) %>%
  group_by(date, test_start, test_end, n_samples) %>%               
  summarise(n_inferred = n(), .groups = "drop")

# Adding 95% CIs to recombinant proportions
ci_exact <- function(k, n) binom.test(k, n)$conf.int

res_two_or_more_parents <- res_two_or_more_parents %>%
  mutate(
    ci_prop_lo  = map2_dbl(n_inferred, n_samples, ~ ci_exact(.x, .y)[1]),
    ci_prop_hi  = map2_dbl(n_inferred, n_samples, ~ ci_exact(.x, .y)[2])
  )

# Extracting window start and end dates
windows <- res_two_or_more_parents %>% select(test_start, test_end)

# Loading in ONS prevalence and averaging within windows
ons <- read.table("../../data/ons_survey_data.tsv")
ons <- ons %>%
  transmute(
    day   = V1, month_name = V2, year = V3, percent = V4,
    prop  = percent / 100,
    date  = as.Date(paste(year, month_name, day, sep = "-"), format = "%Y-%B-%d")
  )

# Modify windows so that we store test_start and test_end as dates
setDT(windows)
windows[, `:=`(test_start = as.Date(test_start), test_end = as.Date(test_end))]

# Add columns start and end, which are set to date for now
setDT(ons)
ons[, `:=`(start = date, end = date)]
setkey(ons, start, end)
setkey(windows, test_start, test_end)

# Match dates within windows and average within windows
ans <- foverlaps(
  x = ons, y = windows,
  by.x = c("start","end"),
  by.y = c("test_start","test_end"),
  type = "within",      # date within [test_start, test_end]
  nomatch = 0L
)[
  , .(avg_prop = mean(prop, na.rm = TRUE),
      n_days  = .N),
  by = .(test_start, test_end)
]

### At this point, we have both the averaged ONS prevalence and detected recombinant proportion, but we also want to adjust the former  ### 

### Organizing dataframes ### 

# Split parents (for those with two parental lineages)
tmp <- res_two_parents %>%
  separate(parental_lineages, into = c("pA", "pB"), sep = ";", remove = FALSE)

# How many rows have "other" as a parent (within those with two parental lineages)
n_with_other <- tmp %>%
  filter(str_to_lower(pA) == "other" | str_to_lower(pB) == "other") %>%
  nrow()

# Drop rows with "other", make consistent ordering, and tally
pair_counts <- tmp %>%
  filter(str_to_lower(pA) != "other", str_to_lower(pB) != "other") %>%
  mutate(parent1 = pmin(pA, pB), parent2 = pmax(pA, pB)) %>%
  count(parent1, parent2, name = "n_sequences") %>%
  arrange(desc(n_sequences), parent1, parent2)

# Version of pair counts during ONS period
pair_counts_during_ons <- tmp %>%
  filter(str_to_lower(pA) != "other", str_to_lower(pB) != "other") %>%
  mutate(parent1 = pmin(pA, pB), parent2 = pmax(pA, pB)) %>%
  filter(test_end <= as.Date("2023-03-19")) %>%
  count(parent1, parent2, name = "n_sequences") %>%
  arrange(desc(n_sequences), parent1, parent2) 

# pair_counts has the number of detected recombinants with each unique parental lineage pair

# Keep "other", make consistent ordering, and tally
pair_counts_w_other <- tmp %>%
  mutate(parent1 = pmin(pA, pB), parent2 = pmax(pA, pB)) %>%
  count(parent1, parent2, test_start, name = "n_sequences") %>%
  arrange(test_start, parent1, parent2)

### Obtain lineage proportions in test windows ### 

# Folder containing lineage frequencies
dir_in <- "../../run-on-cluster/real-data-analysis/output/sliding_windows/expected_recombinant_freq"

# Regex for filenames like: lineage_freq_2024-02-19_2024-02-25.csv.gz
pat <- "^lineage_freq_(\\d{4}-\\d{2}-\\d{2})_(\\d{4}-\\d{2}-\\d{2})\\.csv\\.gz$"

# Get a list of file names corresponding to each test window
file_df <- tibble(
  path = list.files(dir_in, pattern = pat, full.names = TRUE)) %>%
  mutate(
    fname      = basename(path),
    test_start = as.Date(str_match(fname, pat)[, 2]),
    test_end   = as.Date(str_match(fname, pat)[, 3]))

# Get lineage frequencies within each test window
lineage_freq_all <- file_df %>%
  select(path, test_start, test_end) %>%
  pmap_dfr(function(path, test_start, test_end) {
    read_csv(path, show_col_types = FALSE) %>%
      mutate(test_start = test_start,
             test_end   = test_end)})

# Here, we obtain sum (pipj) across all i<j, which is what we need to fit the linear regression later
# Windows can include periods when ONS estimates are not available (because we will left join ONS later)
# We are intentionally including other
pair_mass <- lineage_freq_all %>%                      
  group_by(test_start, test_end) %>%      
  summarise(
    s1 = sum(p, na.rm = TRUE),
    s2 = sum(p^2, na.rm = TRUE),
    sum_pipj = 0.5 * (s1^2 - s2),
    .groups = "drop"
  )

### Using lineage proportions + ONS prev estimates, we next estimate theta and phi using a linear regression ###

# For robust SEs
library(sandwich)
library(lmtest)

# Get number of samples and number of detected recombinants with two parental lineages in each window
res_pairs <- res %>%
  add_count(date, name = "n_samples") %>%
  filter(str_count(parental_lineages, ";") == 1) %>%
  group_by(date, test_start, test_end, n_samples) %>%                
  summarise(n_inferred = n(), .groups = "drop") %>%
  mutate(prop = n_inferred/n_samples,
         ci_prop_lo  = map2_dbl(n_inferred, n_samples, ~ ci_exact(.x, .y)[1]),
         ci_prop_hi  = map2_dbl(n_inferred, n_samples, ~ ci_exact(.x, .y)[2]))

# To the inferred number of recombinant sequences in each window, left join sum pipj in each window and the window-averaged ons prev.
window_stats <- res_pairs %>% left_join(pair_mass) %>% left_join(ans)

# x_w = prev(w) sum pi(w)pj(w)
window_stats$x <- window_stats$avg_prop*window_stats$sum_pipj
# y_w = R^{total}(w)/n(w)
window_stats$y <- window_stats$n_inferred/window_stats$n_samples
# Remove rows where we don't have prevalence
window_stats <- window_stats[!is.na(window_stats$x),]

# Linear regression to est. theta and phi
lm <- lm(y~x, data = window_stats)

# Get summary of lm and calculate HC3 SEs
summary_lm <- summary(lm)

vc_hc3 <- sandwich::vcovHC(lm, type = "HC3")
hc3_test <- coeftest(lm, vcov. = vc_hc3)
p_hc3 <- hc3_test["x", "Pr(>|t|)"]

# Point estimates
phi_hat   <- summary_lm$coefficients[1, 1]
beta_hat  <- summary_lm$coefficients[2, 1]
theta_hat <- phi_hat + beta_hat

# Standard errors using HC3
se_phi   <- sqrt(vc_hc3[1, 1])
se_theta <- sqrt(vc_hc3[1, 1] + vc_hc3[2, 2] + 2 * vc_hc3[1, 2])

# Critical value (e.g., 95% CI)
alpha <- 0.05
z_crit <- qnorm(1 - alpha / 2)

# Wald CIs
phi_CI   <- c(phi_hat - z_crit * se_phi,   phi_hat + z_crit * se_phi)
theta_CI <- c(theta_hat - z_crit * se_theta, theta_hat + z_crit * se_theta)

phi_hat
phi_CI

theta_hat
theta_CI

# Number of observations
n_obs <- nrow(window_stats)

# Caption
cap_text <- bquote(
  hat(theta) == .(round(theta_hat, 3)) ~
    "(95% CI:" ~
    "[" * .(round(theta_CI[1], 3)) * "," * .(round(theta_CI[2], 3)) * "]);" ~
    hat(phi) == .(round(phi_hat, 3)) ~
    "(95% CI:" ~
    "[" * .(round(phi_CI[1], 3)) * "," * .(round(phi_CI[2], 3)) * "]);" ~
    N == .(n_obs)
)

p <- ggplot(window_stats, aes(x, y)) +
  geom_point(alpha = 0.5) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    linewidth = 0.8
  ) +
  labs(
    x = expression(x[w]),
    y = expression(R^{total}(w) / n(w)),
    caption = cap_text
  ) +
  theme_bw(base_size = 15) +
  theme(
    plot.caption = element_text(
      hjust = 0,
      size  = 11    # try 9–11 if needed
    )
  )

p

ggsave(
  filename = "../figs/phi_regression.png",
  plot     = p,
  width    = 6.5,
  height   = 4.5,
  units    = "in",
  dpi      = 300
)

### Finally, we are ready to estimate recombinant counts ###

# Recall that lineage_freq_all contains all lineage frequencies within each window

# Get pipj for all i<j and all windows (using self-join with window id)
pairs_by_window <- lineage_freq_all %>%
  filter(test_end <= as.Date("2023-03-19")) %>%
  transmute(test_start, test_end, lineage = collapsed, p) %>%
  group_by(test_start, test_end) %>%
  mutate(.row = row_number()) %>%
  ungroup() %>%                                       
  inner_join(., ., by = c("test_start", "test_end"),
             suffix = c("_i", "_j")) %>%
  filter(.row_i < .row_j) %>%
  transmute(
    test_start, test_end,
    i_raw  = lineage_i, j_raw  = lineage_j,
    pi_raw = p_i,       pj_raw = p_j
  ) %>%
  mutate(
    i  = pmin(i_raw, j_raw),
    j  = pmax(i_raw, j_raw),
    pi = if_else(i_raw == i,  pi_raw, pj_raw),  
    pj = if_else(i_raw == i,  pj_raw, pi_raw),
    pipj = pi * pj
  ) %>%
  select(test_start, test_end, i, j, pi, pj, pipj)

# Sanity check, this should be the same as pair_mass
test <- pairs_by_window %>% group_by(test_start) %>% summarize(sum(pipj))

# Attach theta, prev, and n
calculate_exp <- pairs_by_window %>% left_join(window_stats %>% select(test_start, test_end, n_samples, avg_prop))
calculate_exp$theta_hat <- theta_hat

# \hat E[R] = theta*n*prev*pi*pj (in each window)
calculate_exp$exp_hat <- calculate_exp$theta_hat * calculate_exp$n_samples * calculate_exp$avg_prop * calculate_exp$pipj

# Sum across windows
calculate_exp_agg <- calculate_exp %>% group_by(i,j) %>% summarize(exp_hat = sum(exp_hat)) %>% arrange(desc(exp_hat))
colnames(calculate_exp_agg) <- c("parent1", "parent2", "exp")

# Sum across lineage pairs
calculate_exp_freq_window <- calculate_exp %>% group_by(test_start, test_end) %>% summarize(exp_hat = sum(exp_hat), n_samples = first(n_samples)) %>% mutate(exp_freq = exp_hat/n_samples)

### Now that we have expected recombinant frequencies in each window, we can plot it against what we detected ###

library(lubridate)
library(scales)
library(patchwork)

res_two_or_more_parents <- res_two_or_more_parents %>% 
  rename(ci_prop_all_lo = ci_prop_lo,
         ci_prop_all_hi = ci_prop_hi)

res_two_or_more_parents$prop_all <- res_two_or_more_parents$n_inferred/res_two_or_more_parents$n_samples

# To detected recombinant proportion, attach the ONS prevalence and expected frequencies 
summary_df <- res_pairs %>% 
  left_join( res_two_or_more_parents %>% select(test_start, prop_all, ci_prop_all_lo, ci_prop_all_hi), by = "test_start" ) %>% 
  left_join( ans %>% select(test_start, ons_prop = avg_prop), by = "test_start" ) %>% 
  left_join( calculate_exp_freq_window %>% select(exp_freq), by = "test_start" )

# Month breaks (using your existing logic)
month_starts <- seq(
  from = as.Date(format(min(summary_df$date, na.rm = TRUE), "%Y-%m-01")),
  to   = as.Date(format(max(summary_df$date, na.rm = TRUE), "%Y-%m-01")),
  by   = "3 months"
)

# Make font larger
theme_set(theme_bw(base_size = 16))

# Calculate correlation between detected recombinant freq. and ONS prevalence estimates
r_raw <- cor(
  summary_df$prop_all,
  summary_df$ons_prop,
  use    = "complete.obs",
  method = "pearson"
)

pearson_test <- cor.test(
  summary_df$prop_all,
  summary_df$ons_prop,
  method = "pearson",
  use = "complete.obs"
)

pearson_test

# Number of windows
n_raw <- sum(complete.cases(summary_df$prop_all, summary_df$ons_prop))

# Regression + HC3
m_raw <- lm(prop_all ~ ons_prop, data = summary_df)
beta_raw <- coef(m_raw)["ons_prop"]

summary(m_raw)

vc_hc3_raw  <- sandwich::vcovHC(m_raw, type = "HC3")
hc3_test_raw <- lmtest::coeftest(m_raw, vcov. = vc_hc3_raw)
p_hc3_raw <- hc3_test_raw["ons_prop", "Pr(>|t|)"]

# Scatterplot
p_sc_raw <- ggplot(summary_df, aes(x = ons_prop, y = prop_all)) +
  geom_point(size = 2, alpha = 0.5, colour = "black") +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0) +
  scale_x_continuous(
    labels = scales::label_percent(accuracy = 0.1),
    name = "ONS prevalence"
  ) +
  scale_y_continuous(
    labels = scales::label_percent(accuracy = 0.1),
    limits = c(0, NA),
    name = "Detected recombinant proportion"
  ) +
  theme_bw(base_size = 16)

# Make plot
p_raw <- ggplot(summary_df, aes(x = date)) +
  geom_ribbon(
    aes(ymin = ci_prop_all_lo, ymax = ci_prop_all_hi),
    alpha = 0.60,
    fill = "grey70",
    colour = NA
  ) +
  geom_line(
    aes(y = prop_all, linetype = "Detected recombinant proportion"),
    linewidth = 0.9,
    colour = "black"
  ) +
  geom_line(
    aes(y = ons_prop, linetype = "ONS prevalence"),
    linewidth = 0.8,
    colour = "black"
  ) +
  scale_linetype_manual(
    values = c(
      "Detected recombinant proportion" = "solid",
      "ONS prevalence" = "twodash"
    ),
    name = NULL
  ) +
  scale_y_continuous(
    labels = scales::label_percent(accuracy = 0.1),
    limits = c(0, NA),
    name = "Proportion"
  ) +
  scale_x_date(
    breaks = month_starts,
    date_labels = "%Y-%m",
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(x = "Date") +
  theme_bw(base_size = 16) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p_raw <- p_raw +
  labs(caption = cap_raw) +
  theme(
    plot.caption = element_text(
      hjust = 1,   # right-aligned
      size  = 10
    ),
    plot.caption.position = "plot"  # anchor to entire plot area
  )

p_ts_raw <- p_raw + labs(caption = NULL)

cap_raw2 <- sprintf(
  paste0(
    "Pearson r = %.2f; ",
    "\u03b2 = %.2f; ",
    "HC3 p-value = %.3g; ",
    "N = %d."
  ),
  r_raw,
  beta_raw,
  p_hc3_raw,
  n_raw
)

p_combined_raw <- (p_ts_raw / p_sc_raw) +
  plot_annotation(
    caption = cap_raw2,
    theme = theme(
      plot.caption = element_text(hjust = 1, size = 14),
      plot.caption.position = "plot"
    )
  )

p_combined_raw

ggsave(
  filename = "../figs/p_combined_raw.png",
  plot     = p_combined_raw,
  width    = 6.5,
  height   = 9,
  units    = "in",
  dpi      = 300
)

### Next, we will plot the Detected recombinant proportion against the adjusted ONS prev. ###

df_adj <- summary_df %>%
  filter(!is.na(prop), !is.na(exp_freq))

# Pearson r (descriptive)
r_adj <- cor(summary_df$prop, summary_df$exp_freq, use = "complete.obs")
n_adj <- nrow(summary_df %>%
                filter(!is.na(prop), !is.na(exp_freq)))

# Regression + HC3
m_adj <- lm(prop ~ exp_freq, data = summary_df)

# Extract slope
beta_adj <- coef(m_adj)["exp_freq"]

# HC3 robust VCOV + significance test
vc_hc3  <- sandwich::vcovHC(m_adj, type = "HC3")
hc3_test <- lmtest::coeftest(m_adj, vcov. = vc_hc3)

# p-value for slope
p_hc3 <- hc3_test["exp_freq", "Pr(>|t|)"]

# Scaling y-axis
k_adj <- max(summary_df$prop, na.rm = TRUE) /
  max(summary_df$exp_freq, na.rm = TRUE)

p_ts_adj <- ggplot(summary_df, aes(x = date)) +
  geom_ribbon(
    aes(ymin = ci_prop_lo, ymax = ci_prop_hi),
    alpha = 0.60,
    fill = "grey70",
    colour = NA
  ) +
  geom_line(
    aes(y = prop, linetype = "Detected recombinant proportion"),
    linewidth = 0.9,
    colour = "black"
  ) +
  geom_line(
    aes(y = exp_freq * k_adj, linetype = "Expected TP proportion"),
    linewidth = 0.8,
    colour = "black"
  ) +
  scale_linetype_manual(
    values = c(
      "Detected recombinant proportion" = "solid",
      "Expected TP proportion" = "twodash"
    ),
    name = NULL
  ) +
  scale_y_continuous(
    labels = scales::label_percent(accuracy = 0.1),
    limits = c(0, NA),
    name = "Detected recombinant proportion",
    sec.axis = sec_axis(
      ~ . / k_adj,
      name = "Expected TP proportion",
      labels = scales::label_percent(accuracy = 0.1)
    )
  ) +
  scale_x_date(
    breaks = month_starts,
    date_labels = "%Y-%m",
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(x = "Date") +
  theme_bw(base_size = 16) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Add scatterplot
p_sc_adj <- ggplot(summary_df, aes(x = exp_freq, y = prop)) +
  geom_point(size = 2, alpha = 0.5, colour = "black") +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0) + 
  scale_x_continuous(
    labels = scales::label_percent(accuracy = 0.1),
    name = "Expected TP proportion"
  ) +
  scale_y_continuous(
    labels = scales::label_percent(accuracy = 0.1),
    limits = c(0, NA),
    name = "Detected recombinant proportion"
  ) +
  theme_bw(base_size = 16)

# Caption (HC3)
cap_adj <- sprintf(
  paste0(
    "Pearson r = %.2f; ",
    "\u03b2 = %.2f; ",
    "HC3 p-value = %.3g; ",
    "N = %d."
  ),
  r_adj,
  beta_adj,
  p_hc3,
  n_adj
)

p_combined <- (p_ts_adj / p_sc_adj) +
  plot_annotation(
    caption = cap_adj,
    theme = theme(
      plot.caption = element_text(hjust = 1, size = 14),
      plot.caption.position = "plot"
    )
  )

p_combined

ggsave(
  filename = "../figs/p_combined.png",
  plot     = p_combined,
  width    = 6.5,
  height   = 9,
  units    = "in",
  dpi      = 300
)

### Finally, we can plot recombinant pairs ###

library(ggrepel)

# Base color
color_blue_group_base   <- "#56B4E9"
color_purple_group_base <- "#CC79A7"

# Define lineage entries
lineage_pairs_blue_group <- c(
  "BA.1-BA.1.1",
  "BA.1-BA.1.17.2",
  "BA.1.1-BA.1.17.2",
  "BA.1.1-BA.2"
)

lineage_pairs_purple_group <- c(
  "BQ.1.1-CH.1.1",
  "CH.1.1-XBB.1.5"
)

selected_lineage_pairs <- c(lineage_pairs_blue_group, lineage_pairs_purple_group)

# Map: lineage pair -> color (same base color within group)
lineage_pair_to_color <- c(
  setNames(rep(color_blue_group_base,   length(lineage_pairs_blue_group)),   lineage_pairs_blue_group),
  setNames(rep(color_purple_group_base, length(lineage_pairs_purple_group)), lineage_pairs_purple_group)
)

# We want to left join while retaining cases when observed pair counts are 0, unlike before
# summary_df_lineages <- calculate_exp_agg %>%
#   left_join(pair_counts_during_ons) %>%
#   filter(parent1 != "other", parent2 != "other") %>%
#   mutate(
#     n_sequences = tidyr::replace_na(n_sequences, 0),
#     
#     # normalize pair order
#     lineage_pair = paste(
#       pmin(parent1, parent2),
#       pmax(parent1, parent2),
#       sep = "-"
#     )
#   )

# What we were doing before
summary_df_lineages <- calculate_exp_agg %>%
  left_join(pair_counts_during_ons) %>%
  filter(parent1 != "other", parent2 != "other") %>%
  filter(!is.na(n_sequences)) %>%
  mutate(
    n_sequences = tidyr::replace_na(n_sequences, 0),
    
    # normalize pair order
    lineage_pair = paste(
      pmin(parent1, parent2),
      pmax(parent1, parent2),
      sep = "-"
    )
  )

# For reference, check how many test sequences during ONS period
res %>% filter(test_end <= as.Date("2023-03-19")) %>% nrow()

# Check that there are no NAs in either column
colSums(is.na(summary_df_lineages[, c("n_sequences", "exp")]))

# Calculate correlation and observation count
r_pairs <- cor(summary_df_lineages$n_sequences, summary_df_lineages$exp, use = "complete.obs")
n_obs   <- nrow(summary_df_lineages)

# Fit OLS and get residuals
fit <- lm(n_sequences ~ exp, data = summary_df_lineages)
slope_hat   <- coef(fit)[["exp"]]     

# Huber–White robust variance 
robust_vcov <- vcovHC(fit, type = "HC3")

# Robust coefficient test
ct_robust <- coeftest(fit, vcov = robust_vcov)
p_val_robust <- ct_robust["exp", "Pr(>|t|)"]

# Add residual values to observations
df_with_res <- summary_df_lineages %>%
  ungroup() %>%
  mutate(
    .resid    = resid(fit),
    .stdresid = rstandard(fit),
    label     = paste0(parent1, " - ", parent2)
  )

# Which points to label? (|standardized residual| >= 3)
pairs_to_label <- df_with_res %>%
  arrange(desc(abs(.resid))) %>%   # sort by extremeness
  slice_head(n = 20)  

cap_text <- sprintf(
  paste0(
    "Pearson r = %.2f; ",
    "β = %.2f; ",
    "HC3 p-value = %.3g; ",
    "N = %d."
  ),
  r_pairs,
  slope_hat,
  p_val_robust,
  n_obs
)

# Plot
p_pairs <- ggplot(
  summary_df_lineages,
  aes(
    x = exp,
    y = n_sequences,
    color = lineage_pair
  )
) +
  geom_point(alpha = 0.7, size = 1.9, stroke = 0) +
  geom_abline(slope = 1, linewidth = 0.6) +
  geom_smooth(
    inherit.aes = FALSE,
    data = summary_df_lineages,
    aes(x = exp, y = n_sequences),
    method = "lm",
    se = FALSE,
    linewidth = 0.8
  ) +
  scale_color_manual(
    values = lineage_pair_to_color,
    breaks = selected_lineage_pairs,   # keeps legend clean
    name   = "Parental lineage pair"
  ) +
  scale_x_continuous(
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  scale_y_continuous(
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    y = "Detected recombinants",
    x = "Expected TP recombinants",
    caption = cap_text
  ) +
  theme_bw(base_size = 15) +
  theme(
    legend.position = "right",
    plot.caption = element_text(hjust = 0)
  )

# Add labels for large standardized residuals (if any)
p_pairs_lab <- p_pairs +
  geom_text_repel(
    data = pairs_to_label,
    aes(label = label),
    size = 3.2,
    max.overlaps = Inf,
    box.padding = 0.25,
    point.padding = 0.2,
    min.segment.length = 0,
    seed = 42
  ) +
  theme(legend.position = "none")

# Draw & save
p_pairs_lab

ggsave(
  filename = "../figs/p_pairs.png",
  plot     = p_pairs_lab,
  width    = 6.5,
  height   = 6.5,
  units    = "in",
  dpi      = 300
)

### Plot underrepresented parental pairs ###

# Define shapes per lineage pair
shapes_for_blue_group   <- c(16, 17, 15, 3)  # 4 distinct shapes
shapes_for_purple_group <- c(16, 17)         # reuse first two shapes

# Map: lineage pair -> shape
lineage_pair_to_shape <- c(
  setNames(shapes_for_blue_group,   lineage_pairs_blue_group),
  setNames(shapes_for_purple_group, lineage_pairs_purple_group)
)

# Modify lineage frequencies
use_plot <- calculate_exp %>%
  mutate(
    # Make sure that keys are in right order
    lineage_pair_key = paste(
      pmin(i, j),
      pmax(i, j),
      sep = "-"
    ),
    
    # Define mid point of test start and end for plotting
    mid_date = test_start + (test_end - test_start) / 2
  ) %>%
  rename(
    key = lineage_pair_key
  )

# Filter data to selected pairs
plot_data_selected_pairs <- use_plot %>%
  filter(key %in% selected_lineage_pairs)

# Plot
expected_joint_frequency_plot <- ggplot(
  plot_data_selected_pairs,
  aes(
    x = mid_date,
    y = pipj,
    color = key,
    shape = key,
    group = key
  )
) +
  geom_line(linewidth = 0.8, alpha = 0.9) +
  geom_point(size = 3.0, alpha = 0.5) +
  scale_color_manual(
    values = lineage_pair_to_color,
    name   = "Parental lineage pair"
  ) +
  scale_shape_manual(
    values = lineage_pair_to_shape,
    name   = "Parental lineage pair"
  ) +
  scale_x_date(
    date_breaks = "2 months",
    date_labels = "%b %Y",
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  scale_y_continuous(
    labels = scales::label_scientific(digits = 2),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    x = "Date",
    y = expression(Joint~lineage~frequencies~(p[i]~p[j]))
  ) +
  theme_bw(base_size = 15) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

expected_joint_frequency_plot

# Save
ggsave(
  filename = "../figs/pipj_selected_pairs.png",
  plot     = expected_joint_frequency_plot,
  width    = 8,
  height   = 4,
  dpi      = 300
)

