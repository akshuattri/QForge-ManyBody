
"""Heisenberg spin chain model."""

import numpy as np
from qforge.physics.operators import pauli_x, pauli_y, pauli_z, kron

def build_heisenberg_hamiltonian(N: int, J_x: float = 1.0, J_y: float = 1.0, 
                                 J_z: float = 1.0, h: float = 0.0, 
                                 bc: str = "open") -> np.ndarray:
    """Build XXZ Heisenberg Hamiltonian."""
    
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    I = np.eye(2, dtype=complex)
    
    n_bonds = N if bc == "periodic" else N - 1
    
    for i in range(n_bonds):
        j = (i + 1) % N
        
        # Build two-site operator
        ops_x, ops_y, ops_z = [], [], []
        for k in range(N):
            if k == i:
                ops_x.append(sx)
                ops_y.append(sy)
                ops_z.append(sz)
            elif k == j:
                ops_x.append(sx)
                ops_y.append(sy)
                ops_z.append(sz)
            else:
                ops_x.append(I)
                ops_y.append(I)
                ops_z.append(I)
        
        if J_x != 0:
            H += J_x * kron(*ops_x)
        if J_y != 0:
            H += J_y * kron(*ops_y)
        if J_z != 0:
            H += J_z * kron(*ops_z)
    
    # Magnetic field
    if h != 0:
        for i in range(N):
            ops = [I]*N
            ops[i] = sz
            H += h * kron(*ops)
    
    H = (H + H.conj().T) / 2
    return H
