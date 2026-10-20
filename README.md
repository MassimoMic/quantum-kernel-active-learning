# Quantum kernels as surrogates in active learning on quantum-chemical labels

Prototype for a master project (Quantum Machine Learning, Ca' Foscari).

## Question

When every training label costs an expensive quantum-chemical calculation,
the *surrogate model* that chooses which molecule to compute next decides how
many calculations are needed. Does a quantum similarity measure drive that
choice better than a classical one?

## Protocol

- **Pool**: ~38 small closed-shell organics (2–4 heavy atoms, C/N/O/F), geometries from RDKit.
- **Labels**: UCCSD-VQE on a fixed active space (4 electrons, 4 orbitals, STO-3G, 8 qubits),
  simulated with PennyLane/lightning. Target = active-space correlation energy
  `E_VQE − E_HF`. Exact diagonalisation is stored as a check on VQE accuracy.
- **Surrogates**: a Gaussian process with one of three Gram matrices:
  1. Tanimoto on Morgan count fingerprints (classical, structural);
  2. RBF on standardised RDKit descriptors (classical, physico-chemical);
  3. Fidelity `|<ψ_i|ψ_j>|²` between the VQE ground states (quantum, descriptor-free).
- **Loop**: 5 random labelled molecules → GP → acquire the most uncertain molecule → repeat to 25.
  10 seeds. Random acquisition as a control.
- **Metric**: RMSE on the still-unlabelled pool after each acquisition — the *labels-to-accuracy* curve —
  not final accuracy.
- **Diagnostics**: spread of off-diagonal kernel values and effective rank, as a first
  concentration check (Thanasilp et al., Nat. Commun. 2024).

A negative result (classical kernels win) is as informative as a positive one.

## Run

```bash
pip install -r requirements.txt
python run_labels.py        # ~40 s per molecule; cached incrementally in data/
python run_al.py            # curves and diagnostics in results/
```

## Caveats (known, deliberate for a prototype)

- The fidelity kernel compares states expressed in each molecule's own
  canonical-orbital basis; overlaps therefore mix physics with basis conventions.
  Orbital localisation or a shared reference basis is the first thing to fix.
- 8-qubit states are trivially simulable; the point here is the *protocol*, not advantage.
- The pool is tiny; conclusions about scaling need larger pools and more qubits.

## Planned extensions

QELM / quantum reservoir surrogates; quantum kernels on classical descriptors
(projected and fidelity); larger active spaces; overlaps via Majorana propagation
(monoprop); DFT labels on QMugs as a classical control.
