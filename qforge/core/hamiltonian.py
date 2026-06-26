"""Base Hamiltonian class."""
import numpy as np
from typing import Tuple, Optional

class BaseHamiltonian:
    """Abstract Hamiltonian."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._matrix = None
    
    def to_dense(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = self._build()
        return self._matrix.copy()
    
    def _build(self) -> np.ndarray:
        raise NotImplementedError
    
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.to_dense())
    
    def eigensystem(self) -> Tuple[np.ndarray, np.ndarray]:
        H = self.to_dense()
        evals, evecs = np.linalg.eigh(H)
        return evals, evecs
    
    def is_hermitian(self, tol: float = 1e-10) -> bool:
        H = self.to_dense()
        return np.allclose(H, H.conj().T, atol=tol)
    
    def is_sparse(self) -> bool:
        return False
