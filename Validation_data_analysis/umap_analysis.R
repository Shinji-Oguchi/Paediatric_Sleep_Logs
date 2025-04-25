"""
UMAP and anomaly detection (Developmental disorders) for Kizugawa pediatric sleep data

Author: Shinji Oguchi
Date: 2025-4-25
"""

library(tidyverse)  # includes dplyr, ggplot2, readr
library(uwot)       # UMAP
library(dbscan)     # HDBSCAN
library(coin)       # Brunner–Munzel test

# 1) Load and join UMAP embeddings with questionnaire data
umap_df <- read_csv("UMAP_Kizugawa_tensor_application_202421.csv") %>% select(-1)
que_df  <- read_csv("Kizugawa_questionnaire_summary_2024129.csv")
kiz_df  <- que_df %>% right_join(umap_df, by = "id")

# 2) Split by age
ages <- 3:5
kiz_list <- map(ages, ~ filter(kiz_df, ay1 == .x))
names(kiz_list) <- paste0("age", ages)

# 3) Density + devdis overlay function
density_devdis_plot <- function(df, age, bins = 6) {
  ggplot(df, aes(x = X, y = Y)) +
    stat_density_2d(geom = "polygon", aes(alpha = ..level..), fill = "green", bins = bins) +
    geom_point(data = filter(df, devdis == 1), aes(x = X, y = Y), color = "red", size = 1) +
    labs(title = paste0("Density_", age, "yr: DevDis in red")) +
    coord_cartesian(xlim = c(-2, 11), ylim = c(0, 12)) +
    theme_classic() + theme(legend.position = "none",
                            axis.text = element_blank(), axis.ticks = element_blank())
}

# 4) Generate density plots for ages 3-5
for (i in seq_along(ages)) {
  age <- ages[i]
  p <- density_devdis_plot(kiz_list[[i]], age, bins = ifelse(age==3, 9, 6))
  ggsave(sprintf("Density_%dyr_devdis.pdf", age), p, width = 5, height = 5)
}

# 5) Distance from centroid calculations
df_distance <- function(df) {
  ctr <- df %>% summarise(cx = mean(X), cy = mean(Y))
  df %>% mutate(
    Distance = sqrt((X - ctr$cx)^2 + (Y - ctr$cy)^2)
  )
}

df_normalize <- function(df) {
  rng <- range(df$Distance, na.rm = TRUE)
  df %>% mutate(
    Normalized_Distance = (Distance - rng[1]) / (rng[2] - rng[1])
  )
}

# Apply to each age group
dist_list <- map(kiz_list, df_distance)
norm_list <- map(dist_list, df_normalize)

# 6) Combine and save distances
dist_all <- bind_rows(norm_list, .id = "age_group")
write_csv(dist_all, "Kizugawa_Distance_from_centroid_norm_3to5.csv")

# 7) Statistical testing: Brunner–Munzel for age 5 group
bm_test <- function(df) {
  bm <- brunner.munzel_test(Distance ~ devdis, data = df)
  return(bm)
}
print(bm_test(dist_list[["age5"]]))

# 8) Barplot of distances by devdis per age group
for (i in seq_along(ages)) {
  age <- ages[i]
  df <- dist_list[[i]] %>% filter(!is.na(devdis)) %>% arrange(desc(Distance))
  df <- df %>% mutate(ID = n():1)
  p <- ggplot(df, aes(x = ID, y = Distance, fill = factor(devdis))) +
    geom_col(width = 1) + coord_flip() +
    scale_fill_manual(values = c("0" = "green", "1" = "red")) +
    labs(title = paste0(age, "yr Distance from centroid")) +
    theme_classic() + theme(legend.position = "none",
                            axis.text = element_blank(), axis.ticks = element_blank())
  ggsave(sprintf("Barplot_%dyr_distance.pdf", age), p, width = 5, height = 4)
}
