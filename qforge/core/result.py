"""Result containers."""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class SolverResult:
    """Generic solver result."""
    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    observables: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    converged: bool = True
    error_estimate: Optional[float] = None
    
    def ground_state_energy(self) -> float:
        return float(np.min(self.eigenvalues))
    
    def excited_energy(self, n: int) -> float:
        sorted_eigs = np.sort(self.eigenvalues)
        if n >= len(sorted_eigs):
            raise IndexError(f"State {n} not available")
        return float(sorted_eigs[n])
    
    def energy_gap(self, n: int = 1) -> float:
        sorted_eigs = np.sort(self.eigenvalues)
        return float(sorted_eigs[n] - sorted_eigs[0])
    
    def ground_state(self) -> np.ndarray:
        if self.eigenvectors is None:
            raise ValueError("Eigenvectors not computed")
        idx = np.argmin(self.eigenvalues)
        return self.eigenvectors[:, idx]
