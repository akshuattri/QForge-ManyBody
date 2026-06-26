"""Hubbard model."""
import numpy as np

def build_hubbard_1d(N: int, t: float = 1.0, U: float = 0.0,
                     mu: float = 0.0, bc: str = "open") -> np.ndarray:
    """1D Hubbard: H = -t Σ_i (c†_i↑ c_{i+1}↑ + h.c.) + U Σ_i n_i↑ n_i↓ - μ Σ_i n_i"""
    
    dim = 4**N  # 2 spins per site
    H = np.zeros((dim, dim), dtype=complex)
    
    # Kinetic energy (simplified for demonstration)
    # Full implementation requires Jordan-Wigner mapping
    
    return H
