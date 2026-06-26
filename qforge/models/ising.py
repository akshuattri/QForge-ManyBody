
"""Transverse-field Ising model."""

import numpy as np
from qforge.physics.operators import pauli_x, pauli_z, kron

def build_ising_hamiltonian(N: int, J: float = 1.0, h: float = 0.0,
                            bc: str = "open") -> np.ndarray:
    """Build transverse-field Ising Hamiltonian."""
    
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    
    sx, sz = pauli_x(), pauli_z()
    I = np.eye(2, dtype=complex)
    
    n_bonds = N if bc == "periodic" else N - 1
    
    for i in range(n_bonds):
        j = (i + 1) % N
        
        # σ_z σ_z interaction
        ops = []
        for k in range(N):
            if k == i or k == j:
                ops.append(sz)
            else:
                ops.append(I)
        H -= J * kron(*ops)
    
    # Transverse field
    if h != 0:
        for i in range(N):
            ops = [I]*N
            ops[i] = sx
            H -= h * kron(*ops)
    
    H = (H + H.conj().T) / 2
    return H
