"""Step 2 - active-learning comparison of surrogates on the cached labels.

    python run_al.py

Outputs:
    results/al_curves.png     RMSE vs number of labels, one curve per kernel
    results/kernel_hist.png   distribution of off-diagonal kernel values
    results/summary.json      diagnostics and area-under-curve per kernel
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkal.pool import build_pool
from qkal.pool_strong import build_strong_pool
from qkal.vqe import load_labels
from qkal.features import (morgan_counts, descriptors, tanimoto_kernel, rbf_kernel,
                           fidelity_kernel, kernel_diagnostics)
from qkal.active_learning import active_learning_curve

N_INIT, N_FINAL, SEEDS = 5, 25, 10

STRONG = "--strong" in sys.argv
DATA, OUT = ("data_strong", "results_strong") if STRONG else ("data", "results")
os.makedirs(OUT, exist_ok=True)
labels, states, states_cheap = load_labels(f"{DATA}/labels.json", f"{DATA}/states.npz")
if STRONG:
    byname = {m.name: m for m in build_strong_pool()}
    mols = [byname[l.name] for l in labels]
else:
    mols = build_pool([l.name for l in labels])
y = np.array([l.e_exact - l.e_hf for l in labels])  # exact active-space correlation energy
gap = np.array([l.e_vqe - l.e_exact for l in labels])
# Cheap scalar available for every molecule at the same cost as the cheap state:
# the energy after CHEAP_STEPS VQE steps, relative to HF.
e_cheap = np.array([[l.e_cheap - l.e_hf] for l in labels])
e_cheap_z = (e_cheap - e_cheap.mean()) / (e_cheap.std() + 1e-12)
X_desc = np.vstack([m.features for m in mols]) if STRONG else descriptors(mols)
have_mp2 = all(np.isfinite(l.e_mp2) for l in labels)
if have_mp2:
    e_mp2 = np.array([[l.e_mp2 - l.e_hf] for l in labels])
    e_mp2_z = (e_mp2 - e_mp2.mean()) / (e_mp2.std() + 1e-12)
N_FINAL = min(N_FINAL, len(y) - 3)  # keep at least 3 unlabelled molecules to score on
N_INIT = min(N_INIT, max(2, N_FINAL // 3))
print(f"{len(y)} molecules, target = exact correlation energy, "
      f"mean {y.mean():+.4f} Ha, std {y.std():.4f} Ha; VQE above exact by {gap.mean()*1e3:.2f} mHa on average (max {gap.max()*1e3:.1f})")

kernels = {
    **({} if STRONG else {"Tanimoto (Morgan FP)": tanimoto_kernel(morgan_counts(mols))}),
    ("RBF (geometry features)" if STRONG else "RBF (RDKit descriptors)"): rbf_kernel(X_desc),
    "Delta-learning (cheap energy only)": rbf_kernel(e_cheap_z),
    "RBF (descriptors + cheap energy)": rbf_kernel(np.hstack([X_desc, 3.0 * e_cheap_z])),
    **({"Delta-learning (MP2 energy only)": rbf_kernel(e_mp2_z)} if have_mp2 else {}),
    "Fidelity (cheap 5-step states)": fidelity_kernel(states_cheap),
    **({"Fidelity (MP2 states, no optimisation)": fidelity_kernel(np.load(f"{DATA}/states.npz")["states_stepmp2"])}
       if "states_stepmp2" in np.load(f"{DATA}/states.npz").files else {}),
    "Fidelity (converged states, oracle)": fidelity_kernel(states),
}
# The last kernel sees the converged state of *unlabelled* molecules, i.e.
# the expensive calculation itself: it is an upper bound, not a usable
# surrogate. It is kept for comparison and drawn dashed.

summary = {}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, K in kernels.items():
    curves = np.array([active_learning_curve(K, y, N_INIT, N_FINAL, seed=s) for s in range(SEEDS)])
    rnd = np.array([active_learning_curve(K, y, N_INIT, N_FINAL, seed=s, strategy="random")
                    for s in range(SEEDS)])
    x = np.arange(N_INIT, N_FINAL + 1)
    m, sd = curves.mean(0), curves.std(0) / np.sqrt(SEEDS)
    ax.plot(x, m * 1e3, label=name, ls="--" if "oracle" in name else (":" if "cheap energy" in name else "-"))
    ax.fill_between(x, (m - sd) * 1e3, (m + sd) * 1e3, alpha=0.2)
    summary[name] = {
        **kernel_diagnostics(K),
        "auc_uncertainty_mHa": float(curves.mean(0).sum() * 1e3),
        "auc_random_mHa": float(rnd.mean(0).sum() * 1e3),
        "rmse_final_mHa": float(m[-1] * 1e3),
    }
    print(f"{name:36s} final RMSE {m[-1]*1e3:6.2f} mHa   "
          f"AUC unc {summary[name]['auc_uncertainty_mHa']:7.1f}  rnd {summary[name]['auc_random_mHa']:7.1f}   "
          f"offdiag {summary[name]['offdiag_mean']:.3f}±{summary[name]['offdiag_std']:.3f}  "
          f"eff.rank {summary[name]['effective_rank']:.1f}")

ax.set_xlabel("labelled molecules (VQE calculations spent)")
ax.set_ylabel("RMSE on unlabelled pool  [mHa]")
ax.set_title("Active learning with uncertainty sampling")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/al_curves.png", dpi=150)

fig, axes = plt.subplots(1, len(kernels), figsize=(3.3 * len(kernels), 3))
for ax, (name, K) in zip(axes, kernels.items()):
    off = K[~np.eye(len(K), dtype=bool)]
    ax.hist(off, bins=30)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("off-diagonal kernel value")
fig.tight_layout()
fig.savefig(f"{OUT}/kernel_hist.png", dpi=150)

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print(f"-> {OUT}/al_curves.png, {OUT}/kernel_hist.png, {OUT}/summary.json")
