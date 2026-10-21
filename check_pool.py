"""Quick sanity check of the whole pool: HF energy of the qubit Hamiltonian
vs PySCF SCF energy, and active-space correlation energy from exact
diagonalisation. No VQE, ~5 s per molecule.

    python check_pool.py
"""
import warnings
warnings.filterwarnings("ignore")
import pennylane as qml
from qkal.pool import build_pool
from qkal.vqe import hamiltonian, exact_energy, scf_energy, ACTIVE_ELECTRONS, N_QUBITS

dev = qml.device("default.qubit", wires=N_QUBITS)
hf = qml.qchem.hf_state(ACTIVE_ELECTRONS, N_QUBITS)
bad = 0
print(f"{'name':26s} {'E_SCF':>12s} {'E_HF(qubit)':>12s} {'Ecorr(exact)':>13s}  ok")
for m in build_pool():
    try:
        H = hamiltonian(m)
    except ValueError as e:
        print(f"{m.name:26s} skipped: {e}"); continue
    @qml.qnode(dev)
    def e():
        qml.BasisState(hf, wires=range(N_QUBITS)); return qml.expval(H)
    e_hf = float(e()); e_scf, conv = scf_energy(m); e_ex = exact_energy(H)
    ok = conv and abs(e_scf - e_hf) < 1e-5 and -0.5 < e_ex - e_hf < 0
    bad += not ok
    print(f"{m.name:26s} {e_scf:12.5f} {e_hf:12.5f} {e_ex-e_hf:+13.4f}  {'OK' if ok else 'BAD'}")
print(f"\n{bad} inconsistent molecules")
