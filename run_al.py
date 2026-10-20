"""Step 2 - active-learning comparison of surrogates on the cached labels.

    python run_al.py

Outputs:
    results/al_curves.png     RMSE vs number of labels, one curve per kernel
    results/kernel_hist.png   distribution of off-diagonal kernel values
    results/summary.json      diagnostics and area-under-curve per kernel
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkal.pool import build_pool
from qkal.vqe import load_labels
from qkal.features import (morgan_counts, descriptors, tanimoto_kernel, rbf_kernel,
                           fidelity_kernel, kernel_diagnostics)
from qkal.active_learning import active_learning_curve

N_INIT, N_FINAL, SEEDS = 3, 9, 10  # prototype values; use 5, 25 on the full pool

labels, states = load_labels("data/labels.json", "data/states.npz")
mols = build_pool([l.name for l in labels])
y = np.array([l.e_vqe - l.e_hf for l in labels])  # active-space correlation energy
print(f"{len(y)} molecules, target = correlation energy, "
      f"mean {y.mean():+.4f} Ha, std {y.std():.4f} Ha")

kernels = {
    "Tanimoto (Morgan FP)": tanimoto_kernel(morgan_counts(mols)),
    "RBF (RDKit descriptors)": rbf_kernel(descriptors(mols)),
    "Fidelity (VQE states)": fidelity_kernel(states),
}

summary = {}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, K in kernels.items():
    curves = np.array([active_learning_curve(K, y, N_INIT, N_FINAL, seed=s) for s in range(SEEDS)])
    rnd = np.array([active_learning_curve(K, y, N_INIT, N_FINAL, seed=s, strategy="random")
                    for s in range(SEEDS)])
    x = np.arange(N_INIT, N_FINAL + 1)
    m, sd = curves.mean(0), curves.std(0) / np.sqrt(SEEDS)
    ax.plot(x, m * 1e3, label=name)
    ax.fill_between(x, (m - sd) * 1e3, (m + sd) * 1e3, alpha=0.2)
    summary[name] = {
        **kernel_diagnostics(K),
        "auc_uncertainty_mHa": float(curves.mean(0).sum() * 1e3),
        "auc_random_mHa": float(rnd.mean(0).sum() * 1e3),
        "rmse_final_mHa": float(m[-1] * 1e3),
    }
    print(f"{name:26s} final RMSE {m[-1]*1e3:6.2f} mHa   "
          f"AUC unc {summary[name]['auc_uncertainty_mHa']:7.1f}  rnd {summary[name]['auc_random_mHa']:7.1f}   "
          f"offdiag {summary[name]['offdiag_mean']:.3f}±{summary[name]['offdiag_std']:.3f}  "
          f"eff.rank {summary[name]['effective_rank']:.1f}")

ax.set_xlabel("labelled molecules (VQE calculations spent)")
ax.set_ylabel("RMSE on unlabelled pool  [mHa]")
ax.set_title("Active learning with uncertainty sampling")
ax.legend()
fig.tight_layout()
fig.savefig("results/al_curves.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, (name, K) in zip(axes, kernels.items()):
    off = K[~np.eye(len(K), dtype=bool)]
    ax.hist(off, bins=30)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("off-diagonal kernel value")
fig.tight_layout()
fig.savefig("results/kernel_hist.png", dpi=150)

json.dump(summary, open("results/summary.json", "w"), indent=2)
print("-> results/al_curves.png, results/kernel_hist.png, results/summary.json")
