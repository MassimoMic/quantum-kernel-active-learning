"""Step 3 - cost scan: how much does each VQE step buy the fidelity kernel?

Requires labels produced with CHECKPOINTS (states_step{k} in data/states.npz).

    python run_scan.py
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qkal.vqe import Label
from qkal.features import fidelity_kernel, kernel_diagnostics
from qkal.active_learning import active_learning_curve

N_INIT, N_FINAL, SEEDS = 5, 25, 10
STRONG = "--strong" in sys.argv
DATA, OUT = ("data_strong", "results_strong") if STRONG else ("data", "results")
os.makedirs(OUT, exist_ok=True)
labels = [Label(**d) for d in json.load(open(f"{DATA}/labels.json"))]
z = np.load(f"{DATA}/states.npz")
y = np.array([l.e_exact - l.e_hf for l in labels])  # exact active-space correlation energy
keys = [k[len("states_step"):] for k in z.files if k.startswith("states_step")]
steps = sorted((int(k) for k in keys if k.isdigit())) + [k for k in keys if not k.isdigit()]
if not steps:
    raise SystemExit("no checkpoint states in data/states.npz: relabel with the current vqe.py")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
summary = {}
for k in steps + ["converged"]:
    S = z["states"] if k == "converged" else z[f"states_step{k}"]
    K = fidelity_kernel(S)
    curves = np.array([active_learning_curve(K, y, N_INIT, N_FINAL, seed=s) for s in range(SEEDS)])
    m = curves.mean(0)
    x = np.arange(N_INIT, N_FINAL + 1)
    lab = {"converged": "converged (oracle)", "mp2": "MP2-initialised state (no optimisation)"}.get(k, f"{k} VQE steps")
    ax.plot(x, m * 1e3, label=lab, ls="--" if k == "converged" else (":" if k == "mp2" else "-"))
    summary[str(k)] = {**kernel_diagnostics(K), "rmse_final_mHa": float(m[-1] * 1e3),
                       "auc_mHa": float(m.sum() * 1e3)}
    print(f"{str(k):>10s} steps  final RMSE {m[-1]*1e3:6.2f} mHa  AUC {m.sum()*1e3:7.1f}  "
          f"offdiag {summary[str(k)]['offdiag_mean']:.3f}±{summary[str(k)]['offdiag_std']:.3f}  "
          f"eff.rank {summary[str(k)]['effective_rank']:.1f}")
ax.set_xlabel("labelled molecules")
ax.set_ylabel("RMSE on unlabelled pool  [mHa]")
ax.set_title("Fidelity kernel vs. cost of the cheap state")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/scan_steps.png", dpi=150)
json.dump(summary, open(f"{OUT}/scan_summary.json", "w"), indent=2)
print(f"-> {OUT}/scan_steps.png, {OUT}/scan_summary.json")
