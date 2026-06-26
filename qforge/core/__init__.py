"""
QForge Core Module

Abstract base classes and data structures for quantum research framework.

Classes:
    - QuantumModel: Abstract quantum model interface
    - QuantumHamiltonian: Abstract Hamiltonian operator
    - QuantumSolver: Abstract solver interface
    - QuantumResult: Result data container
    - QuantumState: State representation
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import time


@dataclass
class QuantumState:
    """Quantum state (wavefunction or density matrix)."""
    vector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    dim: int = 0
    is_pure: bool = True
    
    def normalize(self):
        """Normalize state."""
        if self.vector is not None:
            norm = np.linalg.norm(self.vector)
            self.vector /= norm
        elif self.density_matrix is not None:
            trace = np.trace(self.density_matrix)
            self.density_matrix /= trace
    
    @property
    def purity(self) -> float:
        """Purity Tr(ρ²)."""
        if self.density_matrix is not None:
            return float(np.trace(self.density_matrix @ self.density_matrix).real)
        return 1.0
    
    @property
    def entropy(self) -> float:
        """von Neumann entropy -Tr(ρ ln ρ)."""
        if self.density_matrix is not None:
            eigs = np.linalg.eigvalsh(self.density_matrix)
            eigs = eigs[eigs > 1e-14]
            return float(-np.sum(eigs * np.log(eigs)))
        return 0.0


@dataclass
class QuantumResult:
    """Generic result container."""
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    observables: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = None
    computation_time: float = 0.0
    converged: bool = True
    
    def ground_state_energy(self) -> float:
        return float(np.min(self.eigenvalues))
    
    def energy_gap(self, n: int = 1) -> float:
        sorted_eigs = np.sort(self.eigenvalues)
        return float(sorted_eigs[n] - sorted_eigs[0])


class QuantumModel:
    """Abstract base class for quantum models."""
    
    def __init__(self, system_size: int):
        self.system_size = system_size
        self.parameters = {}
        self.metadata = {}
    
    def set_parameter(self, name: str, value: float):
        """Set model parameter."""
        self.parameters[name] = float(value)
    
    def get_parameter(self, name: str) -> float:
        """Get model parameter."""
        return self.parameters.get(name, 0.0)
    
    def build_hamiltonian(self) -> 'QuantumHamiltonian':
        """Build Hamiltonian (must implement)."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model."""
        return {
            "system_size": self.system_size,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }


class QuantumHamiltonian:
    """Abstract base class for Hamiltonian."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self._matrix = None
    
    def to_dense(self) -> np.ndarray:
        """Return dense matrix."""
        if self._matrix is None:
            self._matrix = self._build_matrix()
        return self._matrix.copy()
    
    def _build_matrix(self) -> np.ndarray:
        """Build Hamiltonian matrix (must implement)."""
        raise NotImplementedError
    
    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """Check Hermiticity."""
        H = self.to_dense()
        return np.allclose(H, H.conj().T, atol=tol)
    
    def eigenvalues(self) -> np.ndarray:
        """Get eigenvalues."""
        return np.linalg.eigvalsh(self.to_dense())
    
    def eigensystem(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get eigenvalues and eigenvectors."""
        H = self.to_dense()
        return np.linalg.eigh(H)


class QuantumSolver:
    """Abstract base class for solvers."""
    
    def __init__(self, rtol: float = 1e-8, atol: float = 1e-10, verbose: bool = False):
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose
        self.last_result = None
    
    def solve(self, hamiltonian: QuantumHamiltonian) -> QuantumResult:
        """Solve for eigensystem (must implement)."""
        raise NotImplementedError
    
    def get_result(self) -> QuantumResult:
        """Get last result."""
        return self.last_result


print("QForge Core module loaded")
