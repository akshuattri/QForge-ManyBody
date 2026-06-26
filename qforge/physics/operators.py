
"""Physics operators: Pauli, Spin, Bosons, Fermions."""

import numpy as np
from typing import List, Tuple

def pauli_x(): return np.array([[0, 1], [1, 0]], dtype=complex)
def pauli_y(): return np.array([[0, -1j], [1j, 0]], dtype=complex)
def pauli_z(): return np.array([[1, 0], [0, -1]], dtype=complex)
def pauli_plus(): return np.array([[0, 1], [0, 0]], dtype=complex)
def pauli_minus(): return np.array([[0, 0], [1, 0]], dtype=complex)

def spin_x(s=0.5): 
    dim = int(2*s + 1)
    S = np.zeros((dim, dim), dtype=complex)
    for i in range(dim-1):
        m = s - i
        coeff = np.sqrt(s*(s+1) - m*(m-1))
        S[i, i+1] = coeff/2
        S[i+1, i] = coeff/2
    return S

def spin_y(s=0.5):
    dim = int(2*s + 1)
    S = np.zeros((dim, dim), dtype=complex)
    for i in range(dim-1):
        m = s - i
        coeff = np.sqrt(s*(s+1) - m*(m-1))
        S[i, i+1] = -1j*coeff/2
        S[i+1, i] = 1j*coeff/2
    return S

def spin_z(s=0.5):
    dim = int(2*s + 1)
    return np.diag([s - i for i in range(dim)], dtype=complex)

def spin_plus(s=0.5):
    dim = int(2*s + 1)
    S = np.zeros((dim, dim), dtype=complex)
    for i in range(dim-1):
        m = s - i
        coeff = np.sqrt(s*(s+1) - m*(m-1))
        S[i, i+1] = coeff
    return S

def spin_minus(s=0.5):
    dim = int(2*s + 1)
    S = np.zeros((dim, dim), dtype=complex)
    for i in range(1, dim):
        m = s - (i-1)
        coeff = np.sqrt(s*(s+1) - m*(m+1))
        S[i, i-1] = coeff
    return S

def bosonic_creation(dim: int) -> np.ndarray:
    """Bosonic creation operator a†."""
    a = np.zeros((dim, dim), dtype=complex)
    for i in range(dim-1):
        a[i, i+1] = np.sqrt(i+1)
    return a

def bosonic_annihilation(dim: int) -> np.ndarray:
    """Bosonic annihilation operator a."""
    a = np.zeros((dim, dim), dtype=complex)
    for i in range(1, dim):
        a[i, i-1] = np.sqrt(i)
    return a

def fermionic_creation(n_sites: int, site: int) -> np.ndarray:
    """Fermionic creation operator c†_i (Jordan-Wigner)."""
    dim = 2**n_sites
    c = np.zeros((dim, dim), dtype=complex)
    for state in range(dim):
        if not (state & (1 << site)):  # Bit not set
            new_state = state | (1 << site)
            parity = bin(state & ((1 << site)-1)).count('1') % 2
            c[new_state, state] = (-1)**parity
    return c

def fermionic_annihilation(n_sites: int, site: int) -> np.ndarray:
    """Fermionic annihilation operator c_i (Jordan-Wigner)."""
    return fermionic_creation(n_sites, site).conj().T

def kron(*ops): 
    """Kronecker product of operators."""
    result = ops[0].astype(complex)
    for op in ops[1:]:
        result = np.kron(result, op.astype(complex))
    return result

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[A, B] = AB - BA"""
    return A @ B - B @ A

def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """{A, B} = AB + BA"""
    return A @ B + B @ A
