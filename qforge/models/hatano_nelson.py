"""Hatano-Nelson non-Hermitian model."""
import numpy as np

def build_hatano_nelson(N: int, alpha: float = 0.1, t: float = 1.0) -> np.ndarray:
    """Non-Hermitian tight-binding chain: H_{i,i+1}=t*e^α, H_{i+1,i}=t*e^{-α} (OBC)."""
    H = np.zeros((N, N), dtype=complex)
    for i in range(N - 1):
        H[i, i + 1] = t * np.exp(-alpha)   # right hopping
        H[i + 1, i] = t * np.exp(alpha)    # left hopping (asymmetric)
    return H
