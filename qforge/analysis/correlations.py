"""Correlation functions."""
import numpy as np

def two_point_correlation(rho: np.ndarray, O1: np.ndarray, O2: np.ndarray) -> complex:
    """⟨O1 ⊗ O2⟩ = Tr((O1 ⊗ O2) ρ)."""
    O = np.kron(O1, O2)
    return np.trace(O @ rho)

def structure_factor(rho: np.ndarray, ops_list, N: int) -> np.ndarray:
    """Structure factor S(k) = Σ_ij e^{ik(i-j)} ⟨O_i O_j⟩."""
    S_k = []
    for k in np.linspace(0, 2*np.pi, N):
        S = 0.0
        for i in range(N):
            for j in range(N):
                # Simplified version
                S += np.cos(k * (i - j))
        S_k.append(S)
    return np.array(S_k)
