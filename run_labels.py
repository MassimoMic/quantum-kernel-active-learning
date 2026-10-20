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
from qkal.vqe import run_vqe, Label

warnings.filterwarnings("ignore")
JSON, NPZ = "data/labels.json", "data/states.npz"

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
names = list(POOL_SMILES)[:limit]

labels, states = [], []
if os.path.exists(JSON):
    labels = [Label(**d) for d in json.load(open(JSON))]
    states = list(np.load(NPZ)["states"])
done = {l.name for l in labels}

for mol in build_pool([n for n in names if n not in done]):
    try:
        lab, psi = run_vqe(mol)
    except ValueError as e:  # e.g. too few orbitals for the active space
        print(f"{mol.name:28s} skipped: {e}")
        continue
    labels.append(lab)
    states.append(psi)
    print(f"{lab.name:28s} Ecorr={lab.e_vqe - lab.e_hf:+.5f} Ha  "
          f"VQE-exact={lab.e_vqe - lab.e_exact:+.2e}  steps={lab.n_steps}  {lab.seconds:5.0f}s",
          flush=True)
    with open(JSON, "w") as f:
        json.dump([l.__dict__ for l in labels], f, indent=2)
    np.savez_compressed(NPZ, states=np.array(states), names=np.array([l.name for l in labels]))

print(f"\n{len(labels)} molecules labelled -> {JSON}, {NPZ}")
