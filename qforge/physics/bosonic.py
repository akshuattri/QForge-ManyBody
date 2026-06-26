"""Bosonic operators."""
import numpy as np

def bosonic_operators(n_levels: int):
    """Creation and annihilation operators."""
    a_dag = np.zeros((n_levels, n_levels), dtype=complex)
    a = np.zeros((n_levels, n_levels), dtype=complex)
    
    for i in range(n_levels-1):
        a_dag[i, i+1] = np.sqrt(i+1)
        a[i+1, i] = np.sqrt(i+1)
    
    return a_dag, a

def number_operator(n_levels: int) -> np.ndarray:
    """Number operator n = a† a."""
    a_dag, a = bosonic_operators(n_levels)
    return a_dag @ a
