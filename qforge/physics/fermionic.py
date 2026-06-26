"""Fermionic operators (Jordan-Wigner)."""
import numpy as np

def fermionic_operators(n_sites: int, site: int):
    """Fermionic creation/annihilation (Jordan-Wigner)."""
    dim = 2**n_sites
    c_dag = np.zeros((dim, dim), dtype=complex)
    c = np.zeros((dim, dim), dtype=complex)
    
    for state in range(dim):
        if not (state & (1 << site)):
            new_state = state | (1 << site)
            parity = bin(state & ((1 << site)-1)).count('1') % 2
            sign = (-1)**parity
            c_dag[new_state, state] = sign
            c[state, new_state] = sign
    
    return c_dag, c

def number_operator(n_sites: int, site: int) -> np.ndarray:
    """Number operator n_i = c†_i c_i."""
    c_dag, c = fermionic_operators(n_sites, site)
    return c_dag @ c
