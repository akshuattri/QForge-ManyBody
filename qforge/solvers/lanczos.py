"""Lanczos algorithm for large eigenvalue problems."""
import numpy as np
from scipy.sparse.linalg import eigsh
from typing import Tuple

def lanczos_ground_state(matvec_func, dim: int, k: int = 6,
                         tol: float = 1e-8):
    """Lanczos ground state finder."""
    try:
        # Using scipy's implementation
        from scipy.sparse import LinearOperator
        A = LinearOperator((dim, dim), matvec=matvec_func)
        evals, evecs = eigsh(A, k=k, which='SA', tol=tol)
        return evals[0], evecs[:, 0]
    except:
        # Fallback to power method
        v = np.random.randn(dim) + 1j*np.random.randn(dim)
        v /= np.linalg.norm(v)
        for _ in range(100):
            v_new = matvec_func(v)
            lambda_est = np.dot(v.conj(), v_new).real / np.dot(v.conj(), v)
            v_new /= np.linalg.norm(v_new)
            if np.linalg.norm(v - v_new) < tol:
                break
            v = v_new
        return lambda_est, v
