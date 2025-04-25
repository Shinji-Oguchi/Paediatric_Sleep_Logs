"""
Correlation heatmap

Author: Shinji Oguchi
Date: 2025-4-25
"""

library(tidyverse)

# 1) Load and prepare bases
art_basis <- read_csv("art_childcare_2024027.csv") %>%
  select(-X1) %>%
  set_names(c("k",
              "Circadian.1","Circadian.2","Circadian.3","Circadian.4",
              "Gap.1","Gap.2","Gap.3","Gap.4",
              "Error.1","Error.2","Error.3"))

# 2) Load sleep info and personal data
art_sleep    <- read_csv("discovery_data_information.csv") %>% select(-X1)
art_personal <- read_csv("social_jetlag.csv") %>%
  select(k, ay1, am1, age, Bf, Sa, fm, ju)

# 3) Merge everything
art_all <- art_basis %>%
  left_join(art_sleep,    by = "k") %>%
  right_join(art_personal, by = "k")
write_csv(art_all, "art_childcare_full_2024024.csv")

# 4) Compute duration quartile by age group
# Define a function that assigns quartiles given thresholds
assign_quartile <- function(total_hour, thresholds) {
  case_when(
    total_hour >= thresholds[1] & total_hour <= thresholds[2] ~ "3",
    total_hour >= thresholds[3] & total_hour <= thresholds[4] ~ "2",
    TRUE ~ "1"
  )
}

# thresholds list by ay1
thresholds_list <- list(
  `1` = c(11,14,10,16),
  `2` = c(11,14,10,16),
  `3` = c(10,13,14,14),
  `4` = c(10,13,14,14),
  `5` = c(10,13,14,14)
)

art_quart <- art_all %>%
  group_by(ay1) %>%
  mutate(
    quartile = assign_quartile(total_hour,
                               thresholds_list[[as.character(unique(ay1))]])
  ) %>%
  ungroup() %>%
  mutate(age = ay1 + am1/12)

write_csv(art_quart, "art_childcare_all_with_quartile.csv")

# 5) Spearman correlation between bases (cols 2–12) and sleep features (cols 13–20)
cor_mat <- art_quart %>%
  select(2:12, 13:20) %>%
  cor(method = "spearman", use = "pairwise.complete.obs") %>%
  round(2)
write_csv(as.data.frame(cor_mat), "spearman_cor_bases_sleep_2024027.csv")

# 6) Scatter plot: Error.2 vs evening_hour
ggplot(art_quart, aes(x = Error.2, y = evening_hour)) +
  geom_point(color = "blue", alpha = 0.6) +
  labs(title = "Error.2 vs Evening Hour",
       x = "Error Component 2",
       y = "Evening Sleep Hour") +
  theme_minimal()

ggsave("scatter_Error2_evening.pdf", width = 5, height = 4)

# 7) Heatmap of correlation matrix
data_long <- cor_mat %>%
  rownames_to_column(var = "Basis") %>%
  pivot_longer(-Basis, names_to = "Feature", values_to = "rho")

ggplot(data_long, aes(x = Feature, y = Basis, fill = rho)) +
  geom_tile() +
  geom_text(aes(label = rho), size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-1,1), name = "Spearman rho") +
  scale_y_discrete(limits = rev) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,
                                   face = "bold"),
        axis.text.y = element_text(face = "bold"))

ggsave("heatmap_bases_sleep_correlation.pdf", width = 6, height = 5)
