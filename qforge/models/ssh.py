
"""SSH (Su-Schrieffer-Heeger) topological model."""

import numpy as np

def build_ssh_hamiltonian(N: int, v: float = 1.0, w: float = 0.5) -> np.ndarray:
    """Build SSH Hamiltonian (1D topological)."""
    
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=complex)
    
    # Intra-cell hopping (strong v)
    for i in range(N):
        H[2*i, 2*i+1] = v
        H[2*i+1, 2*i] = v
    
    # Inter-cell hopping (weak w)
    for i in range(N-1):
        H[2*i+1, 2*(i+1)] = w
        H[2*(i+1), 2*i+1] = w
    
    return H
