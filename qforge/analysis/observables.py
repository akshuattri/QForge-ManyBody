
"""Observable analysis and correlation functions."""

import numpy as np
from typing import Dict, Any

def compute_magnetization(rho: np.ndarray, N: int, direction: str = "z") -> float:
    """Compute magnetization ⟨Σ_i S_i^d⟩."""
    from qforge.physics.operators import pauli_x, pauli_y, pauli_z, kron
    
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    I = np.eye(2, dtype=complex)
    
    if direction == "z":
        op = sz
    elif direction == "x":
        op = sx
    elif direction == "y":
        op = sy
    
    M = 0.0
    for i in range(N):
        ops = [I]*N
        ops[i] = op
        Op = kron(*ops)
        M += np.trace(Op @ rho).real
    
    return M

def compute_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy."""
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-14]
    return float(-np.sum(eigs * np.log(eigs)))

def compute_purity(rho: np.ndarray) -> float:
    """Compute purity Tr(ρ²)."""
    return float(np.trace(rho @ rho).real)

def compute_correlation(rho: np.ndarray, N: int, i: int, j: int, 
                       direction: str = "z") -> float:
    """Compute two-point correlation ⟨S_i^d S_j^d⟩."""
    from qforge.physics.operators import pauli_x, pauli_y, pauli_z, kron
    
    sx, sy, sz = pauli_x(), pauli_y(), pauli_z()
    I = np.eye(2, dtype=complex)
    
    if direction == "z":
        op = sz
    elif direction == "x":
        op = sx
    elif direction == "y":
        op = sy
    
    ops = [I]*N
    ops[i] = op
    ops[j] = op
    Op = kron(*ops)
    
    return float(np.trace(Op @ rho).real)
