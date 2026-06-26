"""Krylov subspace methods."""
import numpy as np

def krylov_time_evolution(H: np.ndarray, psi0: np.ndarray, t: float,
                         order: int = 10) -> np.ndarray:
    """Time evolution via Krylov: exp(-iHt) psi0"""
    dim = len(psi0)
    V = np.zeros((dim, order), dtype=complex)
    H_krylov = np.zeros((order, order), dtype=complex)
    
    # Arnoldi iteration
    v = psi0 / np.linalg.norm(psi0)
    V[:, 0] = v
    
    for j in range(order-1):
        w = H @ V[:, j]
        for i in range(j+1):
            H_krylov[i, j] = np.dot(V[:, i].conj(), w)
            w -= H_krylov[i, j] * V[:, i]
        H_krylov[j+1, j] = np.linalg.norm(w)
        if H_krylov[j+1, j] > 1e-10:
            V[:, j+1] = w / H_krylov[j+1, j]
    
    # Exponentiate small matrix
    U_small = np.linalg.matrix_power(
        np.eye(order) - 1j * t * H_krylov[:order, :order],
        -1
    )
    
    psi = V @ U_small[:, 0] * np.linalg.norm(psi0)
    return psi
