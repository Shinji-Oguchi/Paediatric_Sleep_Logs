"""
Processing of Validation data

Author: Shinji Oguchi
Date: 2025-4-25
"""

library(tidyverse)

# 1) Load and combine yearly sleep logs
sleep_files <- c(
  "kizugawa_sleep_2017_missing.csv",
  "kizugawa_sleep_2018_missing.csv"
)
kiz_sleep <- map_dfr(sleep_files, read_csv)

# 2) Filter to IDs present in questionnaire
kiz_que <- read_csv("kizugawa_questionnaire_summary.csv")
id_list <- kiz_que$id
kiz_sleep <- filter(kiz_sleep, id %in% id_list)

# 3) Split into first and second week
kiz_sleep <- kiz_sleep %>%
  mutate(week = if_else(MesDay <= 7, "pre", "post"))

# 4) Exclude IDs with >=2 missing days per week
kiz_sleep <- kiz_sleep %>%
  group_by(id, week) %>%
  filter(sum(is.na(t1)) < 2) %>%
  ungroup()

# 5) Split by weekday and write out
weekdays <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")

walk(weekdays, function(day) {
  day_df <- filter(kiz_sleep, DoW == day)
  write_csv(day_df,
            sprintf("missing1day_kizugawa_2024129_%s.csv", day))
})

# 6) Save combined filtered dataset
write_csv(kiz_sleep, "missing1day_kizugawa_2024129.csv")

