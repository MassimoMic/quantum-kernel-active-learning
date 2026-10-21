"""Step 1 - compute quantum-chemical labels for every molecule in the pool.

    python run_labels.py            # full pool
    python run_labels.py 10         # first 10 molecules only (quick test)

Results are cached in data/labels.json and data/states.npz; molecules
already present are skipped, so the script can be re-run incrementally.
"""
import json
import os
import sys
import warnings

import numpy as np

from qkal.pool import build_pool, POOL_SMILES
from qkal.pool_strong import build_strong_pool
from qkal.vqe import run_vqe, save_labels, Label

warnings.filterwarnings("ignore")
STRONG = "--strong" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA = "data_strong" if STRONG else "data"
os.makedirs(DATA, exist_ok=True)
JSON, NPZ = f"{DATA}/labels.json", f"{DATA}/states.npz"

limit = int(args[0]) if args else None
pool_all = build_strong_pool() if STRONG else None
names = [m.name for m in pool_all][:limit] if STRONG else list(POOL_SMILES)[:limit]

labels, states, states_cheap = [], [], []
ckpt = {}  # step -> list of states
if os.path.exists(JSON):
    labels = [Label(**d) for d in json.load(open(JSON))]
    z = np.load(NPZ)
    states, states_cheap = list(z["states"]), list(z["states_cheap"])
    ckpt = {k[len("states_step"):]: list(z[k]) for k in z.files if k.startswith("states_step")}
done = {l.name for l in labels}

todo = [m for m in pool_all if m.name in names and m.name not in done] if STRONG \
    else build_pool([n for n in names if n not in done])
for mol in todo:
    try:
        lab, psi, psi_cheap, cps = run_vqe(mol)
    except ValueError as e:  # e.g. too few orbitals for the active space
        print(f"{mol.name:28s} skipped: {e}")
        continue
    labels.append(lab)
    states.append(psi)
    states_cheap.append(psi_cheap)
    for k, v in cps.items():
        ckpt.setdefault(k, []).append(v)
    print(f"{lab.name:28s} Ecorr={lab.e_vqe - lab.e_hf:+.5f} Ha  MP2frac={(lab.e_mp2 - lab.e_hf)/(lab.e_vqe - lab.e_hf):.2f}  "
          f"VQE-exact={lab.e_vqe - lab.e_exact:+.2e}  steps={lab.n_steps}  {lab.seconds:5.0f}s",
          flush=True)
    save_labels(labels, states, states_cheap, JSON, NPZ, ckpt)

print(f"\n{len(labels)} molecules labelled -> {JSON}, {NPZ}")
