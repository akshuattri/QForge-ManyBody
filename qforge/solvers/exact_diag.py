
"""Exact diagonalization solver."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import time

def exact_diagonalization(H: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
    """Diagonalize Hamiltonian exactly."""
    if not np.allclose(H, H.conj().T, atol=1e-10):
        raise ValueError("H must be Hermitian")
    
    dim = H.shape[0]
    if verbose:
        print(f"Diagonalizing {dim}×{dim} Hamiltonian")
    
    t0 = time.time()
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    computation_time = time.time() - t0
    
    if verbose:
        print(f"  E_0 = {eigenvalues[0]:.8f}")
        print(f"  Gap = {eigenvalues[1] - eigenvalues[0]:.8f}")
        print(f"  Time = {computation_time:.3f}s")
    
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "computation_time": computation_time,
        "dim": dim,
        "converged": True,
    }
