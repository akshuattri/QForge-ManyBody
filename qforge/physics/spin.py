"""Spin operators for arbitrary spin."""
import numpy as np

def spin_operators(s: float = 0.5):
    """Generate spin-s operators."""
    dim = int(2*s + 1)
    
    # S_z diagonal
    Sz = np.diag([s - i for i in range(dim)], dtype=complex)
    
    # S_+ and S_-
    Sp = np.zeros((dim, dim), dtype=complex)
    Sm = np.zeros((dim, dim), dtype=complex)
    
    for i in range(dim-1):
        m = s - i
        coeff = np.sqrt(s*(s+1) - m*(m-1))
        Sp[i, i+1] = coeff
        Sm[i+1, i] = coeff
    
    # S_x = (S+ + S-)/2, S_y = (S+ - S-)/(2i)
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2j)
    
    return Sx, Sy, Sz, Sp, Sm
