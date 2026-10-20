"""Quantum-chemical labels for the pool.

For each molecule we build the electronic Hamiltonian in a fixed active
space (4 electrons, 4 orbitals, STO-3G -> 8 qubits), then compute

    E_HF      : Hartree-Fock energy of the active-space Hamiltonian
    E_exact   : exact diagonalisation of the 8-qubit Hamiltonian (CASCI)
    E_VQE     : UCCSD VQE energy
    psi       : final VQE statevector (2**8 complex amplitudes)

The learning target is the active-space correlation energy
    y = E_VQE - E_HF
which is small, chemically meaningful and comparable across molecules.
E_exact tells us how far the VQE is from the true minimum.
"""
from dataclasses import dataclass, asdict
import json
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .pool import Molecule

ACTIVE_ELECTRONS = 4
ACTIVE_ORBITALS = 4
N_QUBITS = 2 * ACTIVE_ORBITALS


@dataclass
class Label:
    name: str
    e_hf: float
    e_exact: float
    e_vqe: float
    n_params: int
    n_steps: int
    seconds: float


def hamiltonian(mol: Molecule):
    m = qml.qchem.Molecule(
        mol.symbols, pnp.array(mol.coords, requires_grad=False),
        charge=mol.charge, basis_name="sto-3g", unit="angstrom",
    )
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        m, method="pyscf",
        active_electrons=ACTIVE_ELECTRONS, active_orbitals=ACTIVE_ORBITALS,
    )
    assert n_qubits == N_QUBITS
    return H


def exact_energy(H) -> float:
    mat = qml.matrix(H, wire_order=range(N_QUBITS))
    return float(np.linalg.eigvalsh(mat)[0])


def run_vqe(mol: Molecule, max_steps: int = 200, lr: float = 0.3, tol: float = 1e-6,
            seed: int = 0):
    t0 = time.time()
    H = hamiltonian(mol)
    hf_state = qml.qchem.hf_state(ACTIVE_ELECTRONS, N_QUBITS)
    singles, doubles = qml.qchem.excitations(ACTIVE_ELECTRONS, N_QUBITS)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    n_params = len(singles) + len(doubles)

    try:
        dev = qml.device("lightning.qubit", wires=N_QUBITS)
        diff = "adjoint"
    except Exception:  # pennylane-lightning not installed
        dev = qml.device("default.qubit", wires=N_QUBITS)
        diff = "best"

    @qml.qnode(dev, diff_method=diff)
    def energy(theta):
        qml.UCCSD(theta, wires=range(N_QUBITS), s_wires=s_wires,
                  d_wires=d_wires, init_state=hf_state)
        return qml.expval(H)

    @qml.qnode(dev)
    def state(theta):
        qml.UCCSD(theta, wires=range(N_QUBITS), s_wires=s_wires,
                  d_wires=d_wires, init_state=hf_state)
        return qml.state()

    rng = np.random.default_rng(seed)
    theta = pnp.array(rng.normal(0, 0.01, n_params), requires_grad=True)
    e_hf = float(energy(pnp.zeros(n_params, requires_grad=False)))

    opt = qml.AdamOptimizer(stepsize=lr)
    prev = e_hf
    steps = 0
    for steps in range(1, max_steps + 1):
        theta, e = opt.step_and_cost(energy, theta)
        if abs(prev - e) < tol:
            break
        prev = e
    e_vqe = float(energy(theta))
    psi = np.asarray(state(theta))
    e_exact = exact_energy(H)

    label = Label(mol.name, e_hf, e_exact, e_vqe, n_params, steps, time.time() - t0)
    return label, psi


def save_labels(labels: list[Label], states: np.ndarray, path_json: str, path_npz: str):
    with open(path_json, "w") as f:
        json.dump([asdict(l) for l in labels], f, indent=2)
    np.savez_compressed(path_npz, states=states, names=np.array([l.name for l in labels]))


def load_labels(path_json: str, path_npz: str):
    with open(path_json) as f:
        labels = [Label(**d) for d in json.load(f)]
    z = np.load(path_npz)
    return labels, z["states"]
