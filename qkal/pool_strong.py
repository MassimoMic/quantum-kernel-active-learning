"""Pool B: geometries where single-reference methods (HF, MP2) break down.

Instead of different molecules at equilibrium, this pool scans *geometries*
of a few molecules into the strongly correlated regime: stretched multiple
bonds and a twisted double bond. Same active space (4 electrons, 4
orbitals, 8 qubits) as pool A, so every script works unchanged.

Each Molecule carries `features` = (family one-hot, stretch ratio r/r_eq,
torsion/90deg), the classical "descriptor" a surrogate is allowed to see.
"""
import numpy as np

from .pool import Molecule

FAMILIES = ["N2", "C2H2", "H2O", "C2H4_twist", "HCN"]


def _mol(name, fam, symbols, coords, ratio=1.0, torsion=0.0):
    onehot = [1.0 if fam == f else 0.0 for f in FAMILIES]
    m = Molecule(name, fam, symbols, np.array(coords, float), 0)
    m.features = np.array(onehot + [ratio, torsion / 90.0])
    return m


def n2(ratio):
    r = 1.098 * ratio
    return _mol(f"N2_r{ratio:.2f}", "N2", ["N", "N"], [[0, 0, 0], [0, 0, r]], ratio)


def c2h2(ratio):
    rcc, rch = 1.203 * ratio, 1.063
    return _mol(f"C2H2_r{ratio:.2f}", "C2H2", ["H", "C", "C", "H"],
                [[0, 0, -rch], [0, 0, 0], [0, 0, rcc], [0, 0, rcc + rch]], ratio)


def h2o(ratio):
    r, ang = 0.958 * ratio, np.deg2rad(104.5 / 2)
    return _mol(f"H2O_r{ratio:.2f}", "H2O", ["O", "H", "H"],
                [[0, 0, 0], [r * np.sin(ang), 0, r * np.cos(ang)],
                 [-r * np.sin(ang), 0, r * np.cos(ang)]], ratio)


def c2h4_twist(deg):
    rcc, rch, a = 1.339, 1.086, np.deg2rad(121.3)
    t = np.deg2rad(deg)
    h = [[rch * np.sin(a), 0, -rch * np.cos(a)], [-rch * np.sin(a), 0, -rch * np.cos(a)]]
    h2 = [[rch * np.sin(a) * np.cos(t), rch * np.sin(a) * np.sin(t), rcc + rch * np.cos(a)],
          [-rch * np.sin(a) * np.cos(t), -rch * np.sin(a) * np.sin(t), rcc + rch * np.cos(a)]]
    return _mol(f"C2H4_t{deg:02d}", "C2H4_twist", ["C", "C", "H", "H", "H", "H"],
                [[0, 0, 0], [0, 0, rcc]] + h + h2, 1.0, deg)


def hcn(ratio):
    rcn, rch = 1.156 * ratio, 1.064
    return _mol(f"HCN_r{ratio:.2f}", "HCN", ["H", "C", "N"],
                [[0, 0, -rch], [0, 0, 0], [0, 0, rcn]], ratio)


def build_strong_pool():
    ratios = [0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3]
    pool = [n2(r) for r in ratios] + [c2h2(r) for r in ratios] + \
           [h2o(r) for r in ratios] + [hcn(r) for r in ratios] + \
           [c2h4_twist(d) for d in (0, 15, 30, 45, 60, 75, 85, 90)]
    return pool
