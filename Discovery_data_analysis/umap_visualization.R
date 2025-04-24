"""
Umap visualization of art childcare pediatric sleep data.

Author: Shinji Oguchi
Date: 2025-4-25
"""

library(tidyverse)  # includes ggplot2, dplyr, etc.
library(uwot)      # UMAP implementation
env <- ls()
library(dbscan)     # HDBSCAN implementation

set.seed(1234)

# 0) Load and filter data
# We already normalized "art_childcare_2024027.csv" and saved it as "15_tensor_umap_minmax_normalized.csv")
ubasis <- read_csv("15_tensor_umap_minmax_normalized.csv") %>%
  filter(ay1 %in% 1:5)
# select features for UMAP
ubasis_1 <- ubasis %>% select(21:35)

# 1) Compute UMAP embedding
fit_umap <- uwot::umap(
  ubasis_1,
  n_neighbors = 20,
  n_components = 2,
  n_trees = 50,
  metric = "euclidean",
  verbose = TRUE
)

# 2) Combine embedding with metadata
u_data_age <- as_tibble(fit_umap) %>%
  set_names(c("V1", "V2")) %>%
  bind_cols(ubasis)

# 3) Plot UMAP colored by age
p_umap <- u_data_age %>%
  ggplot(aes(x = V1, y = V2, color = age)) +
    geom_point(size = 1) +
    scale_color_gradient(low = "blue", high = "red") +
    labs(title = "UMAP_AC_1to5") +
    coord_cartesian(xlim = c(-5, 5), ylim = c(-5, 5)) +
    theme_classic()

ggsave("UMAP_AC_1to5.pdf", p_umap, width = 6, height = 5)

# 4) Density plot function per age
plot_density <- function(data, age_val) {
  data %>%
    filter(ay1 == age_val) %>%
    ggplot(aes(x = V1, y = V2)) +
      stat_density_2d(
        geom = "polygon",
        aes(alpha = ..level..),
        fill = "blue",
        bins = 6
      ) +
      labs(title = paste0("Density_", age_val, "year")) +
      coord_cartesian(xlim = c(-5, 5), ylim = c(-5, 5)) +
      theme_classic() +
      theme(
        legend.position = "none",
        axis.text = element_blank(),
        axis.ticks = element_blank()
      )
}

# generate and save density plots for ages 1:5
for (age in 1:5) {
  p <- plot_density(u_data_age, age)
  ggsave(sprintf("Density_%dyear.pdf", age), p, width = 5, height = 5)
}

# 5) HDBSCAN clustering function per age
plot_hdbscan <- function(data, age_val, min_pts) {
  df <- data %>% filter(ay1 == age_val) %>% select(V1, V2)
  hc <- hdbscan(df, minPts = min_pts)
  df <- df %>%
    mutate(
      cluster = factor(hc$cluster),
      alpha   = ifelse(cluster == 0, 0.1, 1)
    )
  # define colors: cluster 0 gray, others rainbow
  clust_levels <- levels(df$cluster)
  colors <- set_names(
    c("gray", rainbow(length(clust_levels) - 1)),
    clust_levels
  )
  df %>%
    ggplot(aes(x = V1, y = V2, color = cluster, alpha = alpha)) +
      geom_point(size = 1) +
      scale_color_manual(values = colors) +
      labs(title = paste0("HDBSCAN ", age_val, "year minPts", min_pts)) +
      coord_cartesian(xlim = c(-5, 5), ylim = c(-5, 5)) +
      theme_classic() +
      theme(
        legend.position = "none",
        axis.text = element_blank(),
        axis.ticks = element_blank()
      )
}

# specify minPts for each age
min_pts_map <- c(`1` = 40, `2` = 40, `3` = 39, `4` = 40, `5` = 18)

# generate and save HDBSCAN plots
for (age in 1:5) {
  p <- plot_hdbscan(u_data_age, age, min_pts_map[as.character(age)])
  ggsave(
    sprintf("HDBSCAN_%dyear_minPts%d.pdf", age, min_pts_map[as.character(age)]),
    p, width = 6, height = 5
  )
}
