library(ggplot2)
library(dplyr)
library(broom)

df <- read.csv("output/parent_dist_and_detection.csv")

fit <- glm(detected ~ parent_dist, data = df, family = binomial())

summary(fit)

# Prediction grid + 95% CI on the link scale, then transform via plogis
grid <- tibble(parent_dist = seq(min(df$parent_dist), max(df$parent_dist), length.out = 200))
pred <- predict(fit, newdata = grid, type = "response", se.fit = TRUE)

level <- 0.95
crit  <- qnorm((1 + level) / 2)

grid <- grid %>%
  mutate(
    fit_prob = pred$fit,
    lwr      = pred$fit - crit * pred$se.fit,
    upr      = pred$fit + crit * pred$se.fit
  )

# Plot
p <- ggplot(df, aes(parent_dist, detected)) +
  geom_point(alpha = 0.10, size = 1.2) +
  geom_ribbon(data = grid, aes(x = parent_dist, ymin = lwr, ymax = upr), inherit.aes = FALSE, alpha = 0.15) +
  geom_line(data = grid, aes(x = parent_dist, y = fit_prob), linewidth = 0.9) +
  labs(x = "Parental Hamming distance", y = "Sensitivity") +
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw(base_size = 16)

ggsave("figs/sens_vs_parentdist_ggplot.png", p, width = 6, height = 4, dpi = 300)

# (Optional) Odds ratios table to mirror your Python printout
tidy(fit, conf.int = TRUE, exponentiate = TRUE) %>%
  select(term, estimate, conf.low, conf.high)
