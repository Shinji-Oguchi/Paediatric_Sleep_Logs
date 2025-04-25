"""
Umap visualization of art childcare pediatric sleep data.

Author: Shinji Oguchi
Date: 2025-4-25
"""

library(tidyverse)  # ggplot2, dplyr, etc.
library(uwot)       # UMAP implementation
dlibrary(dbscan)     # HDBSCAN clustering

# 1) Load embedding data
art_umap <- read_csv("202421_umap_data_art_childcare.csv") %>% select(-1)

# 2) Density plots for ages 1–5
ages <- 1:5
def_plot_density <- function(df, age) {
  df %>%
    filter(ay1 == age) %>%
    ggplot(aes(x = X, y = Y)) +
    stat_density_2d(geom = "polygon", aes(alpha = ..level..), fill = "blue", bins = 6) +
    labs(title = paste0("Density_", age, "yr")) +
    coord_cartesian(xlim = c(-2, 11), ylim = c(0, 12)) +
    theme_classic() +
    theme(legend.position = "none", axis.text = element_blank(), axis.ticks = element_blank())
}
for (a in ages) {
  p <- def_plot_density(art_umap, a)
  ggsave(sprintf("Density_%dyr.pdf", a), p, width = 5, height = 5)
}

# 3) HDBSCAN plots by age (using precomputed Cluster column)
def_plot_hdbscan <- function(df, age, cols) {
  df %>%
    filter(ay1 == age) %>%
    ggplot(aes(x = X, y = Y, color = factor(Cluster))) +
    geom_point(size = 1, alpha = 0.8) +
    scale_color_manual(values = cols, na.value = "transparent") +
    labs(title = paste0(age, "yr HDBSCAN"), color = "Cluster") +
    coord_cartesian(xlim = c(-2, 11), ylim = c(0, 12)) +
    theme_classic() +
    theme(legend.position = "none", axis.text = element_blank(), axis.ticks = element_blank())
}

color_map <- list(
  c("0"="transparent","1"="red2","2"="darkorange1","3"="green"),
  c("0"="transparent","1"="gold1","2"="goldenrod2","3"="green"),
  c("0"="transparent","1"="springgreen1","2"="springgreen3","3"="green"),
  c("0"="transparent","1"="steelblue1","2"="dodgerblue2","3"="green"),
  c("0"="transparent","1"="darkorchid1","2"="violetred","3"="green")
)
for (i in ages) {
  p <- def_plot_hdbscan(art_umap, i, color_map[[i]])
  ggsave(sprintf("HDBSCAN_%dyr.pdf", i), p, width = 5, height = 5)
}

# 4) Aggregate one-week sleep log and compute average trajectories
sleep_data <- read_csv("14obs_child_sleep_data_age1to5_1228.csv")
sleep_log <- sleep_data %>%
  filter(ay1 %in% ages) %>%
  group_by(k, ay1) %>%
  summarise(across(t1:t48, sum), .groups = 'drop') %>%
  mutate(across(t1:t48, ~ . / 7))

# 5) Plot mean trajectory per cluster and age
def_plot_trajectory <- function(log_df, umap_df, age, cols) {
  df <- log_df %>% filter(ay1 == age) %>%
    left_join(select(umap_df, k, Cluster), by = "k")
  for (cl in unique(df$Cluster[df$Cluster != 0])) {
    tmp <- df %>% filter(Cluster == cl) %>% select(t1:t48) %>%
      summarise(across(everything(), mean)) %>%
      pivot_longer(everything(), names_to = 'time', values_to = 'sleep')
    p <- ggplot(tmp, aes(x = as.integer(str_remove(time, 't')), y = sleep)) +
      geom_line(color = cols[as.character(cl)]) +
      geom_area(fill = cols[as.character(cl)], alpha = 1) +
      coord_cartesian(ylim = c(0, 1)) +
      theme_classic() +
      theme(axis.text = element_blank(), axis.ticks = element_blank(), axis.title = element_blank())
    ggsave(sprintf("SleepTrajectory_%dyr_cluster%d.pdf", age, cl), p, width = 5, height = 3)
  }
}
for (i in ages) {
  def_plot_trajectory(sleep_log, art_umap, i, color_map[[i]])
}
