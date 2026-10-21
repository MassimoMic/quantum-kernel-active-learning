"""Candidate pool: small closed-shell molecules with 2-4 heavy atoms.

Every molecule is treated in the same active space (4 electrons in 4
spatial orbitals -> 8 qubits), so that VQE statevectors live in the same
Hilbert space and a fidelity kernel between them is well defined.
Geometries come from RDKit (ETKDG + MMFF94).
"""
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Deliberately a "series": saturated / unsaturated C, N, O, F skeletons.
POOL_SMILES = {
    "methane": "C",
    "ammonia": "N",
    "water": "O",
    "ethane": "CC",
    "ethene": "C=C",
    "ethyne": "C#C",
    "methanol": "CO",
    "methylamine": "CN",
    "fluoromethane": "CF",
    "formaldehyde": "C=O",
    "methanimine": "C=N",
    "hydrogen_cyanide": "C#N",
    "nitrogen": "N#N",
    "carbon_monoxide": "[C-]#[O+]",
    "hydrogen_peroxide": "OO",
    "hydrazine": "NN",
    "hydroxylamine": "NO",
    "propane": "CCC",
    "propene": "C=CC",
    "propyne": "C#CC",
    "acetaldehyde": "CC=O",
    "ethanol": "CCO",
    "ethylamine": "CCN",
    "fluoroethane": "CCF",
    "dimethyl_ether": "COC",
    "dimethylamine": "CNC",
    "acetonitrile": "CC#N",
    "formic_acid": "OC=O",
    "formamide": "NC=O",
    "difluoromethane": "FCF",
    "methanediol": "OCO",
    "ketene": "C=C=O",
    "allene": "C=C=C",
    "isocyanic_acid": "N=C=O",
    "carbon_dioxide": "O=C=O",
    "ozone_like_fluorine_oxide": "FOF",
    "acetic_acid": "CC(=O)O",
    "acetone": "CC(=O)C",
}


@dataclass
class Molecule:
    name: str
    smiles: str
    symbols: list
    coords: np.ndarray  # Angstrom, shape (n_atoms, 3)
    charge: int = 0
    features: np.ndarray | None = None  # geometry features for pool B


def embed(smiles: str, seed: int = 7) -> tuple[list, np.ndarray]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"embedding failed for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return symbols, coords


def build_pool(names: list | None = None) -> list[Molecule]:
    names = names or list(POOL_SMILES)
    out = []
    for n in names:
        smi = POOL_SMILES[n]
        symbols, coords = embed(smi)
        charge = Chem.GetFormalCharge(Chem.MolFromSmiles(smi))
        out.append(Molecule(n, smi, symbols, coords, charge))
    return out
