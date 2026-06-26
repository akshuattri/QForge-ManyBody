"""Fidelity measures."""
import numpy as np

def fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Fidelity F(ρ1, ρ2) = Tr(√(√ρ1 ρ2 √ρ1))."""
    sqrt_rho1 = np.linalg.cholesky(rho1 + 1e-14*np.eye(len(rho1)))
    A = sqrt_rho1 @ rho2 @ sqrt_rho1.conj().T
    eigs = np.linalg.eigvalsh(A)
    eigs = np.maximum(eigs, 0)
    return float(np.trace(np.sqrt(A)))

def purity(rho: np.ndarray) -> float:
    """Purity P = Tr(ρ²)."""
    return float(np.trace(rho @ rho).real)
