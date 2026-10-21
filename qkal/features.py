"""Molecular representations and kernels.

Three similarity measures are compared on the same molecules:

    tanimoto  : Tanimoto on Morgan count fingerprints (classical, structural)
    rbf       : RBF on standardised RDKit descriptors (classical, physico-chemical)
    fidelity  : |<psi_i|psi_j>|^2 between VQE ground states (quantum, no descriptors)

All three return a Gram matrix over the pool so that the same Gaussian
process code can consume any of them.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator

from .pool import Molecule

DESCRIPTOR_NAMES = [
    "MolWt", "NumValenceElectrons", "HeavyAtomCount", "NumHDonors",
    "NumHAcceptors", "TPSA", "MolLogP", "MolMR", "FractionCSP3",
    "NumRotatableBonds", "RingCount", "NumHeteroatoms", "BalabanJ",
    "BertzCT", "Chi0v", "Chi1v", "Kappa1", "Kappa2", "LabuteASA",
]


def morgan_counts(mols: list[Molecule], radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = []
    for m in mols:
        rd = Chem.MolFromSmiles(m.smiles)
        X.append(np.array(gen.GetCountFingerprintAsNumPy(rd), dtype=float))
    return np.vstack(X)


def descriptors(mols: list[Molecule]) -> np.ndarray:
    fns = dict(Descriptors.descList)
    X = []
    for m in mols:
        rd = Chem.MolFromSmiles(m.smiles)
        X.append([fns[n](rd) for n in DESCRIPTOR_NAMES])
    X = np.array(X, dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X


# ---- kernels ---------------------------------------------------------------

def tanimoto_kernel(X: np.ndarray) -> np.ndarray:
    """Tanimoto (min/max) similarity on count vectors."""
    n = len(X)
    K = np.empty((n, n))
    for i in range(n):
        mn = np.minimum(X[i], X).sum(1)
        mx = np.maximum(X[i], X).sum(1)
        K[i] = np.where(mx > 0, mn / mx, 1.0)
    return K


def rbf_kernel(X: np.ndarray, gamma: float | None = None) -> np.ndarray:
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    if gamma is None:  # median heuristic
        gamma = 1.0 / np.median(d2[d2 > 0])
    return np.exp(-gamma * d2)


def fidelity_kernel(states: np.ndarray) -> np.ndarray:
    """|<psi_i|psi_j>|^2 for statevectors stored as rows."""
    G = states.conj() @ states.T
    return np.abs(G) ** 2


def kernel_diagnostics(K: np.ndarray) -> dict:
    """Concentration diagnostics: off-diagonal spread and effective rank."""
    off = K[~np.eye(len(K), dtype=bool)]
    eig = np.linalg.eigvalsh(K)
    eig = eig[eig > 1e-12]
    p = eig / eig.sum()
    return {
        "offdiag_mean": float(off.mean()),
        "offdiag_std": float(off.std()),
        "offdiag_min": float(off.min()),
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
    }
