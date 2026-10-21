# Quantum kernels as surrogates in active learning on quantum-chemical labels

Prototype for a master project in Quantum Machine Learning (Ca' Foscari University, Venice).

## Question

When every training label costs an expensive quantum-chemical calculation, the
*surrogate model* that chooses which molecule to compute next decides how many
calculations are needed. Does a quantum similarity measure between molecular
states drive that choice better than a classical one — and better than the
cheap classical proxies chemistry already has?

## Protocol

- **Pool A** (`qkal/pool.py`): 38 small closed-shell organics (2–4 heavy atoms,
  C/N/O/F), geometries from RDKit (ETKDG + MMFF; not ab-initio equilibria).
- **Pool B** (`qkal/pool_strong.py`): 44 geometries along bond stretches
  (N₂, C₂H₂, H₂O, HCN, 0.9–2.3 × r_eq) and the ethylene torsion (0–90°),
  i.e. the regime where single-reference methods break down.
- **Active space**: 4 electrons in 4 orbitals, STO-3G → 8 qubits, built with PennyLane/PySCF.
  Every label passes three consistency checks (qubit HF = PySCF SCF, UCCSD(θ=0) = HF,
  VQE ≥ exact) before it is stored.
- **Label**: exact active-space correlation energy `E_exact − E_HF` (CASCI by diagonalisation).
  A UCCSD-VQE (Adam, lightning.qubit) produces the *states*; its gap to the exact energy is reported.
- **Surrogates**: a Gaussian process with one of the following Gram matrices.

  | Surrogate | Input | Cost of the input |
  |---|---|---|
  | Tanimoto / Morgan FP | SMILES | ~0 |
  | RBF / RDKit descriptors (pool A) or geometry features (pool B) | SMILES / geometry | ~0 |
  | Δ-learning on the MP2 energy | `E_MP2 − E_HF` (scalar) | one MP2 (ms) |
  | Fidelity between **MP2-initialised UCCSD states** | state vector | one MP2 + one circuit |
  | Fidelity between states after *k* VQE steps (k = 1, 2, 5, 10, 20) | state vector | k/≈85 of a label |
  | Fidelity between **converged VQE states** (*oracle*: uses the label itself) | state vector | the full label |

- **Loop**: 5 random labelled molecules → GP → acquire the most uncertain → repeat to 25;
  10 seeds; random acquisition as control.
- **Metric**: RMSE on the still-unlabelled pool after each acquisition (*labels-to-accuracy*),
  not final accuracy.
- **Diagnostics**: spread of off-diagonal kernel values and effective rank
  (first concentration check, after Thanasilp et al., Nat. Commun. 2024).

## Results on pool A (38 molecules, 25 labels, 10 seeds)

| Surrogate | final RMSE [mHa] | AUC uncertainty | AUC random | off-diag mean ± std | eff. rank |
|---|---|---|---|---|---|
| Tanimoto (Morgan FP) | 12.4 | 520 | 740 | 0.05 ± 0.08 | 33.4 |
| RBF (RDKit descriptors) | 13.8 | 557 | 647 | 0.38 ± 0.22 | 8.9 |
| RBF (descriptors + 5-step energy) | 11.6 | 341 | 595 | 0.40 ± 0.23 | 8.2 |
| Fidelity, 5-step VQE states | 5.3 | 261 | 577 | 0.79 ± 0.13 | 2.6 |
| Fidelity, converged states (oracle) | 3.9 | 197 | 477 | 0.95 ± 0.05 | 1.3 |
| Fidelity, MP2-initialised states | 1.6 | 189 | 568 | 0.99 ± 0.01 | 1.0 |
| **Δ-learning on the MP2 energy (scalar)** | **0.9** | **102** | 1834 | 0.44 ± 0.39 | 3.8 |

Cost scan of the fidelity kernel (final RMSE, mHa): 1 step 22.2 · 2 steps 10.6 ·
5 steps 5.3 · 10 steps 10.1 · 20 steps 9.2 · MP2 state 1.6 · converged 3.9.

### What this says

1. **A cheap classical proxy wins.** A GP on the single number `E_MP2 − E_HF` reaches
   0.9 mHa with 25 labels — better than every quantum-state kernel, including the oracle.
   On closed-shell molecules near equilibrium the correlation energy is ~0.58 × the MP2
   estimate (spread 0.46–0.77), and the GP learns the rest.
2. **The state kernels are, in effect, one number.** Fidelity kernels between MP2 or
   converged states have effective rank ≈ 1: the overlap is dominated by how far each
   state sits from HF, i.e. by the correlation energy itself. They carry no structural
   information beyond the scalar on this pool.
3. **Cheminformatic descriptors are the wrong classical baseline for a quantum-chemical
   target.** Morgan fingerprints and RDKit descriptors do not encode what decides the
   label (multiple bonds, near-degeneracies); comparing a quantum kernel against them
   is how an apparent "quantum advantage" appears.
4. **States along the optimiser trajectory are not a cost knob.** Accuracy vs number of
   VQE steps is non-monotonic (5 steps beat 10 and 20): it reflects Adam's dynamics,
   not chemistry. The MP2 state is the right deterministic cheap state.
5. **Uncertainty sampling works for every surrogate**, and matters most for the 1-D
   MP2 model (AUC 102 vs 1834 random), which extrapolates badly until the extremes
   are labelled.

Concentration is not the bottleneck at 8 qubits and 38 molecules (the most
concentrated kernels perform best); it will be at larger sizes.

## Open question → pool B

MP2 is a near-perfect proxy exactly where nobody needs a quantum computer. Pool B
scans into strong correlation (stretched N₂, twisted ethylene) to test whether the
MP2 scalar still suffices there, or whether a kernel on the state carries information
the scalar does not. Results pending.

## Run

```bash
pip install -r requirements.txt
python check_pool.py                 # HF/active-space sanity check, no VQE
python run_labels.py                 # pool A labels, ~40 s/molecule, cached incrementally in data/
python run_al.py                     # comparison → results/
python run_scan.py                   # cost scan → results/
python run_labels.py --strong        # pool B → data_strong/
python run_al.py --strong            # → results_strong/
python run_scan.py --strong
```

## Caveats

- 8-qubit states are trivially simulable: the subject is the *protocol*, not advantage.
- Fidelities compare states in each molecule's own canonical-orbital basis; overlaps mix
  physics with basis conventions. Orbital localisation or a shared reference is the
  obvious fix if state kernels ever look promising.
- Pool A geometries come from a force field; N₂ in particular is far from its ab-initio
  bond length. Pool B uses explicit internal coordinates.
- Small pools, one active-space size; conclusions about scaling need more.

## Planned extensions

QELM / quantum reservoir surrogates; projected quantum kernels on classical descriptors;
larger active spaces; overlaps via Majorana propagation (monoprop); DFT labels on QMugs
as a classical control.
