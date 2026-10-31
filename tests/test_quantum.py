"""Subspace XY-QAOA must match a full 2^n Qiskit statevector exactly."""
import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate, DiagonalGate
from qiskit.quantum_info import Statevector

sys.path.insert(0, ".")
from qadv import quantum as qm


def full_reference(n, K, Efull, gammas, betas, pairs):
    vec = np.zeros(2 ** n, complex)
    for c in itertools.combinations(range(n), K):
        vec[sum(1 << i for i in c)] = 1
    sv = Statevector(vec / np.linalg.norm(vec))
    qc = QuantumCircuit(n)
    for g, b in zip(gammas, betas):
        qc.append(DiagonalGate(list(np.exp(-1j * g * Efull))), range(n))
        for i, j in pairs:
            qc.append(XXPlusYYGate(2 * b, 0.0), [i, j])   # exp(-i b (XX+YY)/2)
    return sv.evolve(qc).data


def test(trials=6):
    rng = np.random.default_rng(1)
    for mixer in ("ring", "clique"):
        for _ in range(trials):
            n, K = int(rng.integers(4, 8)), None
            K = int(rng.integers(1, n))
            sub = qm.Subspace(n, K, mixer)
            Esub = rng.normal(size=sub.dim)
            Efull = np.zeros(2 ** n)
            Efull[sub.masks] = Esub
            p = int(rng.integers(1, 4))
            g, b = rng.uniform(-1, 1, p), rng.uniform(-1, 1, p)
            ref = full_reference(n, K, Efull, g, b, sub.pairs)
            out = sub.evolve(Esub, g, b)
            assert np.allclose(np.abs(ref[sub.masks]) ** 2, np.abs(out) ** 2, atol=1e-10)
            assert np.allclose(ref[sub.masks], out, atol=1e-10), "amplitudes differ"
            leak = 1 - (np.abs(ref[sub.masks]) ** 2).sum()
            assert leak < 1e-10, f"amplitude left the K-subspace: {leak}"
    print(f"quantum: subspace == Qiskit full statevector, {2 * trials} random circuits, "
          "both mixers, zero leakage -- OK")


if __name__ == "__main__":
    test()
