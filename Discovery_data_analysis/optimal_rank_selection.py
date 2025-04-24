"""
Tensor decomposition and reconstruction error analysis
for undersampled pediatric sleep logs.

Author: Shinji Oguchi
Date: 2025-4-25
"""

import numpy as np
import pandas as pd
import tensorly as tl
import tensorly.decomposition as tsd
from tensorly.decomposition import non_negative_parafac
from tensorly.cp_tensor import cp_to_tensor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from tqdm import tqdm

# 1) Load undersampled data for each weekday
tensor_mon = pd.read_csv("undersampling_Mon_1228.csv")
tensor_tue = pd.read_csv("undersampling_Tue_1228.csv")
tensor_wed = pd.read_csv("undersampling_Wed_1228.csv")
tensor_thu = pd.read_csv("undersampling_Thu_1228.csv")
tensor_fri = pd.read_csv("undersampling_Fri_1228.csv")
tensor_sat = pd.read_csv("undersampling_Sat_1228.csv")
tensor_sun = pd.read_csv("undersampling_Sun_1228.csv")

# 2) Build 3D tensor [subjects x timepoints x weekdays]
df_tensor = tl.tensor(np.zeros((3540, 48, 7)))
df_tensor[..., 0] = tensor_mon.loc[:, "t1":"t48"]
df_tensor[..., 1] = tensor_tue.loc[:, "t1":"t48"]
df_tensor[..., 2] = tensor_wed.loc[:, "t1":"t48"]
df_tensor[..., 3] = tensor_thu.loc[:, "t1":"t48"]
df_tensor[..., 4] = tensor_fri.loc[:, "t1":"t48"]
df_tensor[..., 5] = tensor_sat.loc[:, "t1":"t48"]
df_tensor[..., 6] = tensor_sun.loc[:, "t1":"t48"]

# 3) Compute squared day-to-day differences → gap tensor
dfs = [tensor_mon, tensor_tue, tensor_wed, tensor_thu,
       tensor_fri, tensor_sat, tensor_sun]
day_diffs = []
for i in range(7):
    current = dfs[i].loc[:, "t1":"t48"]
    nextday = dfs[(i+1) % 7].loc[:, "t1":"t48"]
    squared_diff = (current - nextday) ** 2
    day_diffs.append(squared_diff.values)
df_gap = tl.tensor(np.stack(day_diffs, axis=2))

# 4) Define divergence metrics and masked-loss function
def kl_divergence(p, q):
    return entropy(p.ravel(), q.ravel())

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p.ravel(), m.ravel()) + entropy(q.ravel(), m.ravel()))

def calc_loss_with_random_mask(data, rank_max=7, n_rep=10,
                               n_iter_max=1000, random_state=1234,
                               mask_ratio=0.3):
    err_summary = {
        'mse': [[] for _ in range(rank_max)],
        'kl':  [[] for _ in range(rank_max)],
        'js':  [[] for _ in range(rank_max)]
    }
    rng = np.random.default_rng(random_state)
    nonzeros = np.array(np.nonzero(data)).T
    mask_count = int(len(nonzeros) * mask_ratio)

    for _ in tqdm(range(n_rep), desc="Repetitions"):
        selected = rng.choice(len(nonzeros), size=mask_count, replace=False)
        mask = np.zeros_like(data, dtype=bool)
        coords = nonzeros[selected]
        mask[coords[:,0], coords[:,1], coords[:,2]] = True

        for rank in range(1, rank_max+1):
            factors = non_negative_parafac(
                tl.tensor(data),
                rank,
                mask=mask,
                n_iter_max=n_iter_max,
                init='random',
                random_state=random_state
            )
            recon = cp_to_tensor(factors)

            mse = np.sqrt(np.mean((data - recon)**2))
            err_summary['mse'][rank-1].append(mse)

            D = np.where(data > 0, data, 1e-10)
            R = np.where(recon > 0, recon, 1e-10)

            err_summary['kl'][rank-1].append(kl_divergence(D, R))
            err_summary['js'][rank-1].append(js_divergence(D, R))

    return err_summary

# 5) Compute and plot errors for the original tensor
err_summary = calc_loss_with_random_mask(df_tensor)
ranks = range(1, 8)

# extract mean values
mse_vals = [np.mean(err_summary['mse'][i]) for i in range(7)]
kl_vals  = [np.mean(err_summary['kl'][i])  for i in range(7)]
js_vals  = [np.mean(err_summary['js'][i])  for i in range(7)]

# plot line charts
plt.figure(figsize=(10,4))
plt.plot(ranks, mse_vals, marker='o')
plt.title('Original Tensor: MSE by Rank')
plt.xlabel('Rank'); plt.ylabel('MSE'); plt.grid(True)
plt.savefig("orig_MSE_by_rank.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.plot(ranks, kl_vals, marker='x')
plt.title('Original Tensor: KL Divergence by Rank')
plt.xlabel('Rank'); plt.ylabel('KL Divergence'); plt.grid(True)
plt.savefig("orig_KL_by_rank.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.plot(ranks, js_vals, marker='s')
plt.title('Original Tensor: JS Divergence by Rank')
plt.xlabel('Rank'); plt.ylabel('JS Divergence'); plt.grid(True)
plt.savefig("orig_JS_by_rank.pdf"); plt.close()

# plot boxplots
plt.figure(figsize=(10,4))
plt.boxplot(err_summary['mse'], positions=ranks, showfliers=False)
plt.title('Original Tensor: MSE Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('MSE'); plt.grid(True)
plt.savefig("orig_MSE_boxplot.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.boxplot(err_summary['kl'], positions=ranks, showfliers=False)
plt.title('Original Tensor: KL Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('KL Divergence'); plt.grid(True)
plt.savefig("orig_KL_boxplot.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.boxplot(err_summary['js'], positions=ranks, showfliers=False)
plt.title('Original Tensor: JS Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('JS Divergence'); plt.grid(True)
plt.savefig("orig_JS_boxplot.pdf"); plt.close()

# 6) Build reconstruction-error tensor and analyze
factors4 = non_negative_parafac(tl.tensor(df_tensor), rank=4,
                                n_iter_max=1000, random_state=1234)
recon4 = cp_to_tensor(factors4)
df_err = (df_tensor - recon4) ** 2

err_summary2 = calc_loss_with_random_mask(df_err)

# plot recon-error metrics
mse2 = [np.mean(err_summary2['mse'][i]) for i in range(7)]
kl2  = [np.mean(err_summary2['kl'][i])  for i in range(7)]
js2  = [np.mean(err_summary2['js'][i])  for i in range(7)]

plt.figure(figsize=(10,4))
plt.plot(ranks, mse2, marker='o')
plt.title('Reconstruction Error: MSE by Rank')
plt.xlabel('Rank'); plt.ylabel('MSE'); plt.grid(True)
plt.savefig("recon_MSE_by_rank.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.plot(ranks, kl2, marker='x')
plt.title('Reconstruction Error: KL Divergence by Rank')
plt.xlabel('Rank'); plt.ylabel('KL Divergence'); plt.grid(True)
plt.savefig("recon_KL_by_rank.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.plot(ranks, js2, marker='s')
plt.title('Reconstruction Error: JS Divergence by Rank')
plt.xlabel('Rank'); plt.ylabel('JS Divergence'); plt.grid(True)
plt.savefig("recon_JS_by_rank.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.boxplot(err_summary2['mse'], positions=ranks, showfliers=False)
plt.title('Reconstruction Error: MSE Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('MSE'); plt.grid(True)
plt.savefig("recon_MSE_boxplot.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.boxplot(err_summary2['kl'], positions=ranks, showfliers=False)
plt.title('Reconstruction Error: KL Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('KL Divergence'); plt.grid(True)
plt.savefig("recon_KL_boxplot.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.boxplot(err_summary2['js'], positions=ranks, showfliers=False)
plt.title('Reconstruction Error: JS Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('JS Divergence'); plt.grid(True)
plt.savefig("recon_JS_boxplot.pdf"); plt.close()

# 7) Analyze gap tensor errors
err_summary3 = calc_loss_with_random_mask(df_gap)

mse3 = [np.mean(err_summary3['mse'][i]) for i in range(7)]
kl3  = [np.mean(err_summary3['kl'][i])  for i in range(7)]
js3  = [np.mean(err_summary3['js'][i])  for i in range(7)]

plt.figure(figsize=(10,4))
plt.plot(ranks, mse3, marker='o')
plt.title('Gap Tensor: MSE by Rank')
plt.xlabel('Rank'); plt.ylabel('MSE'); plt.grid(True)
plt.savefig("gap_MSE_by_rank.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.plot(ranks, kl3, marker='x')
plt.title('Gap Tensor: KL Divergence by Rank')
plt.xlabel('Rank'); plt.ylabel('KL Divergence'); plt.grid(True)
plt.savefig("gap_KL_by_rank.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.plot(ranks, js3, marker='s')
plt.title('Gap Tensor: JS Divergence by Rank')
plt.xlabel('Rank'); plt.ylabel('JS Divergence'); plt.grid(True)
plt.savefig("gap_JS_by_rank.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.boxplot(err_summary3['mse'], positions=ranks, showfliers=False)
plt.title('Gap Tensor: MSE Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('MSE'); plt.grid(True)
plt.savefig("gap_MSE_boxplot.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.boxplot(err_summary3['kl'], positions=ranks, showfliers=False)
plt.title('Gap Tensor: KL Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('KL Divergence'); plt.grid(True)
plt.savefig("gap_KL_boxplot.pdf"); plt.close()

plt.figure(figsize=(10,4))
plt.boxplot(err_summary3['js'], positions=ranks, showfliers=False)
plt.title('Gap Tensor: JS Distribution by Rank')
plt.xlabel('Rank'); plt.ylabel('JS Divergence'); plt.grid(True)
plt.savefig("gap_JS_boxplot.pdf"); plt.close()

# 8) CP decomposition helper
def decomposition(data, rank, prefix, cmap):
    factors = non_negative_parafac(data, rank=rank,
                                  n_iter_max=1000, random_state=1234)
    weights, [subjects, times, weeks] = factors

    weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    pd.DataFrame(weeks, index=weekdays).to_csv(f"{prefix}_weekday.tsv", sep="\t")

    timepoints = [f"{h/2:.1f}" for h in range(times.shape[0])]
    pd.DataFrame(times, index=timepoints).to_csv(f"{prefix}_time.tsv", sep="\t")

    max_vals = []
    for i in range(rank):
        mat = np.outer(weeks[:,i], times[:,i])
        max_vals.append(mat.max())

    scores = pd.DataFrame(subjects * max_vals,
                          columns=[f"Comp-{i+1}" for i in range(rank)])
    for i in range(rank):
        basis = np.outer(weeks[:,i], times[:,i]) / max_vals[i]
        df_b = pd.DataFrame(basis, index=weekdays, columns=timepoints)
        df_b.to_csv(f"{prefix}_basis_{i+1}.tsv", sep="\t")
        plt.figure(figsize=(6,4))
        sns.heatmap(df_b, cmap=cmap)
        plt.title(f"{prefix} Basis {i+1}")
        plt.tight_layout()
        plt.savefig(f"{prefix}_basis_{i+1}.pdf")
        plt.close()

    return scores

# 9) Run decompositions and save combined subject scores
scores_circ = decomposition(df_tensor, rank=5, prefix="circadian", cmap="Reds")
scores_err  = decomposition(df_err,    rank=4, prefix="error",    cmap="Blues")
scores_gap  = decomposition(df_gap,    rank=3, prefix="gap",      cmap="Greens")

combined = pd.concat([scores_circ, scores_err, scores_gap], axis=1)
combined.to_csv("combined_subject_scores.tsv", sep="\t")
