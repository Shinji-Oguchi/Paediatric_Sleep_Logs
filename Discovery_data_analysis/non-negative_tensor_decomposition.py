"""
Full tensor application using 14 days of art childcare pediatric sleep data.

Author: Shinji Oguchi
Date: 2025-4-25
"""

import numpy as np
import pandas as pd
import tensorly as tl
import tensorly.decomposition as tsd
from tensorly.cp_tensor import cp_to_tensor

# 1) Load one-week undersampled sleep data (Monday–Sunday)
tensor_mon = pd.read_csv("undersampling_Mon_1228.csv")
tensor_tue = pd.read_csv("undersampling_Tue_1228.csv")
tensor_wed = pd.read_csv("undersampling_Wed_1228.csv")
tensor_thu = pd.read_csv("undersampling_Thu_1228.csv")
tensor_fri = pd.read_csv("undersampling_Fri_1228.csv")
tensor_sat = pd.read_csv("undersampling_Sat_1228.csv")
tensor_sun = pd.read_csv("undersampling_Sun_1228.csv")

# 2) Build 3D tensor [subjects x timepoints x weekdays]
df_tensor = tl.tensor(np.zeros((3540, 48, 7)))
df_tensor[..., 0] = tensor_mon.loc[:, "t1":"t48"].values
df_tensor[..., 1] = tensor_tue.loc[:, "t1":"t48"].values
df_tensor[..., 2] = tensor_wed.loc[:, "t1":"t48"].values
df_tensor[..., 3] = tensor_thu.loc[:, "t1":"t48"].values
df_tensor[..., 4] = tensor_fri.loc[:, "t1":"t48"].values
df_tensor[..., 5] = tensor_sat.loc[:, "t1":"t48"].values
df_tensor[..., 6] = tensor_sun.loc[:, "t1":"t48"].values

# 3) CP decomposition (rank=4) for circadian factors
res_cir = tsd.non_negative_parafac(df_tensor,
                                  rank=4,
                                  n_iter_max=1000,
                                  random_state=1234)

# 4) Reconstruction-error tensor
df_recon = cp_to_tensor(res_cir)
df_err = (df_tensor - df_recon) ** 2

# 5) Build squared day-to-day difference tensor (gap tensor)
dfs = [tensor_mon, tensor_tue, tensor_wed,
       tensor_thu, tensor_fri, tensor_sat, tensor_sun]
day_diffs = []
for i in range(7):
    curr = dfs[i].loc[:, "t1":"t48"]
    nxt  = dfs[(i + 1) % 7].loc[:, "t1":"t48"]
    day_diffs.append(((curr - nxt) ** 2).values)
df_gap = tl.tensor(np.stack(day_diffs, axis=2))

# 6) Load 14-day child sleep dataset (ages 1–5)
art_data = pd.read_csv("14obs_child_sleep_data_age1to5_1228.csv")

# split by weekday (2=Mon,...,1=Sun)
art_mon = art_data[art_data["wday"] == 2].reset_index(drop=True)
art_tue = art_data[art_data["wday"] == 3].reset_index(drop=True)
art_wed = art_data[art_data["wday"] == 4].reset_index(drop=True)
art_thu = art_data[art_data["wday"] == 5].reset_index(drop=True)
art_fri = art_data[art_data["wday"] == 6].reset_index(drop=True)
art_sat = art_data[art_data["wday"] == 7].reset_index(drop=True)
art_sun = art_data[art_data["wday"] == 1].reset_index(drop=True)

# 7) Build 3D tensor for child data [obs x timepoints x weekdays]
n_obs = art_mon.shape[0]
art_tensor = tl.tensor(np.zeros((n_obs, 48, 7)))
for i, df_day in enumerate([art_mon, art_tue, art_wed,
                            art_thu, art_fri, art_sat, art_sun]):
    art_tensor[..., i] = df_day.loc[:, "t1":"t48"].values

# 8) Split art_tensor into segments for joint CP fitting
seg1 = art_tensor[0:3540]
seg2 = art_tensor[3540:7080]
seg3 = art_tensor[7080:10620]
seg4 = art_tensor[7934:11474]

# 9) Move subject axis to last dimension
tensor_list = [df_tensor, seg1, seg2, seg3, seg4]
tensor_list = [tl.moveaxis(x, 0, -1) for x in tensor_list]

# 10) Joint CP: initialize on undersampled tensor, then fix modes
init_res = tsd.non_negative_parafac(tensor_list[0],
                                   rank=4,
                                   n_iter_max=1000,
                                   random_state=1234)
results = [init_res]
for seg in tensor_list[1:]:
    res_i = tsd.non_negative_parafac(seg,
                                     rank=4,
                                     init=init_res,
                                     fixed_modes=[0,1],
                                     n_iter_max=1000,
                                     random_state=1234,
                                     mask=None,
                                     verbose=False)
    results.append(res_i)

# 11) Extract and concatenate circadian subject factors
child_factors = []
for idx, res in enumerate(results[1:], start=1):
    mat = res[1][2]
    if idx == 4:
        mat = mat[2686:3540]
    child_factors.append(mat)
child_concat = np.concatenate(child_factors, axis=0)
child_df = pd.DataFrame(child_concat,
                        columns=[f"CircadianComp{i+1}" for i in range(4)])
child_df.to_csv("2024027_Circadian_tensor_application_art_childcare.csv", index=False)

# 12) Build total reconstruction for art segments
recons = [cp_to_tensor(r) for r in results[1:]]
recons[3] = recons[3][..., 2686:3540]
total = tl.concatenate(recons, axis=2, casting="same_kind")
total = tl.moveaxis(total, 2, 0)

# 13) Compute art reconstruction-error tensor
art_err = (art_tensor - total) ** 2

# 14) Joint CP on art_err (rank=3)
err_list = [df_err,
            art_err[0:3540], art_err[3540:7080],
            art_err[7080:10620], art_err[7934:11474]]
err_list = [tl.moveaxis(x, 0, -1) for x in err_list]
init_err = tsd.non_negative_parafac(err_list[0],
                                    rank=3,
                                    n_iter_max=1000,
                                    random_state=1234)
err_results = [init_err]
for seg in err_list[1:]:
    r = tsd.non_negative_parafac(seg,
                                 rank=3,
                                 init=init_err,
                                 fixed_modes=[0,1],
                                 n_iter_max=1000,
                                 random_state=1234,
                                 mask=None,
                                 verbose=False)
    err_results.append(r)

# extract and save error subject factors
child_err = []
for idx, res in enumerate(err_results[1:], start=1):
    m = res[1][2]
    if idx == 4:
        m = m[2686:3540]
    child_err.append(m)
err_concat = np.concatenate(child_err, axis=0)
err_df = pd.DataFrame(err_concat,
                      columns=[f"ErrorComp{i+1}" for i in range(3)])
err_df.to_csv("2024027_Error_tensor_application_art_childcare.csv", index=False)

# 15) Joint CP on gap tensor (rank=4)
gap1 = df_gap
gap2 = art_err[0:3540]  # use art_gap splits instead if defined
gap3 = art_err[3540:7080]
gap4 = art_err[7080:10620]
gap5 = art_err[7934:11474]
# correct variables: replace art_err with art_gap
# Gap splits
gap2 = art_gap[0:3540]
gap3 = art_gap[3540:7080]
gap4 = art_gap[7080:10620]
gap5 = art_gap[7934:11474]

gap_list = [gap1, gap2, gap3, gap4, gap5]
gap_list = [tl.moveaxis(x, 0, -1) for x in gap_list]
init_gap = tsd.non_negative_parafac(gap_list[0],
                                    rank=4,
                                    n_iter_max=1000,
                                    random_state=1234)
gap_results = [init_gap]
for seg in gap_list[1:]:
    rg = tsd.non_negative_parafac(seg,
                                  rank=4,
                                  init=init_gap,
                                  fixed_modes=[0,1],
                                  n_iter_max=1000,
                                  random_state=1234,
                                  mask=None,
                                  verbose=False)
    gap_results.append(rg)

child_gap = []
for idx, res in enumerate(gap_results[1:], start=1):
    g = res[1][2]
    if idx == 4:
        g = g[2686:3540]
    child_gap.append(g)
gap_concat = np.concatenate(child_gap, axis=0)
gap_df = pd.DataFrame(gap_concat,
                      columns=[f"GapComp{i+1}" for i in range(4)])
gap_df.to_csv("2024027_Gap_tensor_application_art_childcare.csv", index=False)

# 16) Combine all bases and save final CSV
cir_bases = pd.read_csv("2024027_Circadian_tensor_application_art_childcare.csv")
gap_bases = pd.read_csv("2024027_Gap_tensor_application_art_childcare.csv")
err_bases = pd.read_csv("2024027_Error_tensor_application_art_childcare.csv")

# preserve subject IDs
ids = art_mon[["k"]]
art_bases = pd.concat([ids, cir_bases, gap_bases, err_bases], axis=1)
art_bases.to_csv("art_childcare_2024027.csv", index=False)
