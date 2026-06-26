"""Entanglement and thermodynamic entropy."""
import numpy as np

def von_neumann_entropy(rho: np.ndarray) -> float:
    """Entanglement entropy S = -Tr(ρ ln ρ)."""
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-14]
    return float(-np.sum(eigs * np.log(eigs)))

def renyi_entropy(rho: np.ndarray, n: int = 2) -> float:
    """Renyi entropy S_n = 1/(1-n) ln(Tr(ρ^n))."""
    if n == 1:
        return von_neumann_entropy(rho)
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-14]
    return float(np.log(np.sum(eigs**n)) / (1 - n))

def mutual_information(rho_AB: np.ndarray, dim_A: int) -> float:
    """Mutual information I(A:B)."""
    # Trace out B to get ρ_A
    rho_A = np.trace(rho_AB.reshape(dim_A, -1, dim_A, -1), axis1=1, axis2=3)
    # Trace out A to get ρ_B
    dim_B = rho_AB.shape[0] // dim_A
    rho_B = np.trace(rho_AB.reshape(dim_A, -1, dim_A, -1), axis1=0, axis2=2)
    
    S_A = von_neumann_entropy(rho_A)
    S_B = von_neumann_entropy(rho_B)
    S_AB = von_neumann_entropy(rho_AB)
    
    return S_A + S_B - S_AB
