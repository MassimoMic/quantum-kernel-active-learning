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

from pool import build_pool, POOL_SMILES
from vqe import run_vqe, save_labels, Label

warnings.filterwarnings("ignore")
JSON, NPZ = "data/labels.json", "data/states.npz"

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
names = list(POOL_SMILES)[:limit]

labels, states, states_cheap = [], [], []
if os.path.exists(JSON):
    labels = [Label(**d) for d in json.load(open(JSON))]
    z = np.load(NPZ)
    states, states_cheap = list(z["states"]), list(z["states_cheap"])
done = {l.name for l in labels}

for mol in build_pool([n for n in names if n not in done]):
    try:
        lab, psi, psi_cheap = run_vqe(mol)
    except ValueError as e:  # e.g. too few orbitals for the active space
        print(f"{mol.name:28s} skipped: {e}")
        continue
    labels.append(lab)
    states.append(psi)
    states_cheap.append(psi_cheap)
    print(f"{lab.name:28s} Ecorr={lab.e_vqe - lab.e_hf:+.5f} Ha  "
          f"VQE-exact={lab.e_vqe - lab.e_exact:+.2e}  steps={lab.n_steps}  {lab.seconds:5.0f}s",
          flush=True)
    save_labels(labels, states, states_cheap, JSON, NPZ)

print(f"\n{len(labels)} molecules labelled -> {JSON}, {NPZ}")
