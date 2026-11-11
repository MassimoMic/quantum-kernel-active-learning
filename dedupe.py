"""Keep the last occurrence of each molecule in the label cache."""
import json, sys, shutil
import numpy as np

DATA = "data_strong" if "--strong" in sys.argv else "data"
JSON, NPZ = f"{DATA}/labels.json", f"{DATA}/states.npz"

labels = json.load(open(JSON))
z = np.load(NPZ)
last = {}
for i, l in enumerate(labels):
    last[l["name"]] = i            # later entries overwrite earlier ones
keep = sorted(last.values())
print(f"{len(labels)} entries -> {len(keep)} unique molecules")

shutil.copy(JSON, JSON + ".bak")
shutil.copy(NPZ, NPZ + ".bak")

json.dump([labels[i] for i in keep], open(JSON, "w"), indent=1)
arrays = {k: z[k][keep] for k in z.files if len(z[k]) == len(labels)}
others = {k: z[k] for k in z.files if len(z[k]) != len(labels)}
if others:
    print("WARNING: arrays not aligned with labels, left untouched:", list(others))
np.savez(NPZ, **arrays, **others)
print("saved; backups in", JSON + ".bak", NPZ + ".bak")
