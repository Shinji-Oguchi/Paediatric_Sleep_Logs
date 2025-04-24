"""
Tensor analysis combining undersampled data and Kizugawa missing-1-day pediatric sleep data.

Author: Shinji Oguchi
Date: 2025-4-25
"""

import numpy as np
import pandas as pd
import tensorly as tl
import tensorly.decomposition as tsd
from tensorly.cp_tensor import cp_to_tensor

# 1) Load one-week undersampled data (Monday–Sunday)
undersampling_files = [
    "undersampling_Mon_1228.csv",
    "undersampling_Tue_1228.csv",
    "undersampling_Wed_1228.csv",
    "undersampling_Thu_1228.csv",
    "undersampling_Fri_1228.csv",
    "undersampling_Sat_1228.csv",
    "undersampling_Sun_1228.csv"
]
undersampling_dfs = [pd.read_csv(f) for f in undersampling_files]

# 2) Build original tensor [3540 x 48 x 7]
df_tensor = tl.tensor(np.zeros((3540, 48, 7)))
for i, df in enumerate(undersampling_dfs):
    df_tensor[..., i] = df.loc[:, "t1":"t48"].values

# 3) Compute original reconstruction-error tensor
tmp_res = tsd.non_negative_parafac(
    df_tensor, rank=4, n_iter_max=1000, random_state=1234
)
df_recon = cp_to_tensor(tmp_res)
df_err = (df_tensor - df_recon) ** 2

# 4) Compute original gap tensor
gap_list = []
for i in range(7):
    curr = undersampling_dfs[i].loc[:, "t1":"t48"]
    nxt  = undersampling_dfs[(i + 1) % 7].loc[:, "t1":"t48"]
    gap_list.append(((curr - nxt) ** 2).values)
df_gap = tl.tensor(np.stack(gap_list, axis=2))

# 5) Load Kizugawa missing-1-day data (1435 subjects)
kiz_files = [
    "missing1day_Kizugawa_2024129_Mon.csv",
    "missing1day_Kizugawa_2024129_Tue.csv",
    "missing1day_Kizugawa_2024129_Wed.csv",
    "missing1day_Kizugawa_2024129_Thu.csv",
    "missing1day_Kizugawa_2024129_Fri.csv",
    "missing1day_Kizugawa_2024129_Sat.csv",
    "missing1day_Kizugawa_2024129_Sun.csv"
]
kiz_dfs = [pd.read_csv(f) for f in kiz_files]

# 6) Build Kiz tensor [1435 x 48 x 7]
kiz_tensor = tl.tensor(np.zeros((1435, 48, 7)))
for i, df in enumerate(kiz_dfs):
    kiz_tensor[..., i] = df.loc[:, "t1":"t48"].values

# 7) Compute Kiz gap tensor
kiz_gap_list = []
for i in range(7):
    curr = kiz_dfs[i].loc[:, "t1":"t48"]
    nxt  = kiz_dfs[(i + 1) % 7].loc[:, "t1":"t48"]
    kiz_gap_list.append(((curr - nxt) ** 2).values)
kiz_gap = tl.tensor(np.stack(kiz_gap_list, axis=2))

# 8) Pad Kiz tensor to match 3540 subjects
pad_count = 3540 - kiz_tensor.shape[0]
blank_tensor = tl.tensor(np.zeros((pad_count, 48, 7)))
kiz_padded = tl.concatenate((kiz_tensor, blank_tensor), axis=0)

# 9) Move subject axis to last before CP
df_tensor_m = tl.moveaxis(df_tensor, 0, -1)
df_err_m    = tl.moveaxis(df_err,    0, -1)
df_gap_m    = tl.moveaxis(df_gap,    0, -1)
kiz_m       = tl.moveaxis(kiz_padded,0, -1)
kiz_gap_m   = tl.moveaxis(kiz_gap,   0, -1)

# 10) CP on original tensor to get base factors
dec_base = tsd.non_negative_parafac(
    df_tensor_m, rank=4, n_iter_max=1000, random_state=1234
)

# 11) Apply CP on Kiz data with missing-mask for circadian patterns
mask_kiz   = ~np.isnan(kiz_m)
kiz_filled = np.nan_to_num(kiz_m)
dec_kiz    = tsd.non_negative_parafac(
    kiz_filled, rank=4,
    init=dec_base,
    fixed_modes=[0,1],
    mask=mask_kiz,
    random_state=1234,
    verbose=False
)

# Extract Kiz circadian components (first 1435 subjects)
child_cir = dec_kiz[1][2][:1435, :]
child_cir_df = pd.DataFrame(
    child_cir,
    columns=[f"Circadian-{i+1}" for i in range(4)]
)

# 12) Compute and CP on Kiz reconstruction-error
dec_recon = cp_to_tensor(dec_kiz)
kiz_recon = tl.moveaxis(dec_recon, 2, 0)[:1435]
kiz_err2  = (kiz_tensor - kiz_recon) ** 2

# Pad and moveaxis for CP
kiz_err_padded = tl.concatenate((kiz_err2, blank_tensor), axis=0)
kiz_err_m      = tl.moveaxis(kiz_err_padded, 0, -1)

# CP on original error tensor to init
dec_err_base = tsd.non_negative_parafac(
    df_err_m, rank=3, n_iter_max=1000, random_state=1234
)

# CP on Kiz error with mask
mask_err   = ~np.isnan(kiz_err_m)
kiz_err_filled = np.nan_to_num(kiz_err_m)
dec_kiz_err    = tsd.non_negative_parafac(
    kiz_err_filled, rank=3,
    init=dec_err_base,
    fixed_modes=[0,1],
    mask=mask_err,
    random_state=1234,
    verbose=False
)
# Extract error components
child_err = dec_kiz_err[1][2][:1435, :]
child_err_df = pd.DataFrame(
    child_err,
    columns=[f"Error-{i+1}" for i in range(3)]
)

# 13) CP on original gap tensor to init
dec_gap_base = tsd.non_negative_parafac(
    df_gap_m, rank=4, n_iter_max=1000, random_state=1234
)
# CP on Kiz gap\mask_gap   = ~np.isnan(kiz_gap_m)
kiz_gap_filled = np.nan_to_num(kiz_gap_m)
dec_kiz_gap    = tsd.non_negative_parafac(
    kiz_gap_filled, rank=4,
    init=dec_gap_base,
    fixed_modes=[0,1],
    mask=mask_gap,
    random_state=1234,
    verbose=False
)
# Extract gap components
child_gap = dec_kiz_gap[1][2][:1435, :]
child_gap_df = pd.DataFrame(
    child_gap,
    columns=[f"Gap-{i+1}" for i in range(4)]
)

# 14) Combine IDs and all components, then save
ids = kiz_dfs[0]["id"]
kiz_all = pd.concat([
    ids,
    child_cir_df,
    child_gap_df,
    child_err_df
], axis=1)
kiz_all.to_csv("kizugawa_all_bases_2024129.csv", index=False)
