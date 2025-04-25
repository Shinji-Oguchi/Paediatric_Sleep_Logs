"""
Undersample Discovery data by age group and weekday.

Author: Shinji Oguchi
Date: 2025-4-25
"""

import numpy as np
import pandas as pd

# 1) Load the full 14-day pediatric sleep dataset (ages 1–5)
df_all = pd.read_csv("14obs_child_sleep_data_age1to5_1228.csv.csv")

# 2) Split data by age and save individual CSV files
dfs = {}
for age in range(1, 6):
    df_age = df_all[df_all['ay1'] == age].copy()
    df_age.to_csv(f"14obs_child_sleep_data_age{age}_1228.csv", index=False)
    dfs[age] = df_age

# 3) Randomly undersample ages 1–4 to 354 unique subject IDs
np.random.seed(1234)
for age in range(1, 5):
    unique_ids = dfs[age]['k'].unique()
    selected = np.random.choice(unique_ids, size=354, replace=False)
    dfs[age] = dfs[age][dfs[age]['k'].isin(selected)].copy()

# 4) Keep age 5 full (no undersampling)
dfs[5] = dfs[5]

# 5) Map weekday codes to English names
weekday_map = {
    1: "Sun",
    2: "Mon",
    3: "Tue",
    4: "Wed",
    5: "Thu",
    6: "Fri",
    7: "Sat"
}

# 6) Generate and save undersampled CSVs by weekday
for wday, name in weekday_map.items():
    # Collect all age groups for the given weekday
    weekly_parts = [
        dfs[age][dfs[age]['wday'] == wday]
        for age in range(1, 6)
    ]
    df_week = pd.concat(weekly_parts, ignore_index=True)
    df_week.to_csv(f"undersampling_{name}_1228.csv", index=False)

# End of script
