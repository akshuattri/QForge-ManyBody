"""Kitaev chain (1D topological superconductor)."""
import numpy as np

def build_kitaev(N: int, t: float = 1.0, Delta: float = 0.5,
                 mu: float = 0.0) -> np.ndarray:
    """Kitaev chain: H = -t Σ (c†_i c_{i+1} + h.c.) - μ Σ n_i + Δ Σ (c_i c_{i+1} + h.c.)"""
    
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    
    # Kinetic hopping
    for i in range(N-1):
        for state in range(dim):
            if (state & (1 << i)) and not (state & (1 << (i+1))):
                new_state = (state ^ (1 << i)) | (1 << (i+1))
                parity = bin(state & ((1 << (i+1))-1)).count('1') % 2
                H[new_state, state] -= t * (-1)**parity
                H[state, new_state] -= t * (-1)**parity
    
    # Chemical potential
    for i in range(N):
        for state in range(dim):
            if state & (1 << i):
                H[state, state] -= mu
    
    return H
