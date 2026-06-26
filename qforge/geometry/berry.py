
"""Quantum geometry: Berry curvature, Chern numbers, etc."""

import numpy as np
from typing import Tuple, Optional

def berry_curvature_2band(k_grid: np.ndarray, H_func) -> np.ndarray:
    """Compute Berry curvature for 2-band model."""
    
    nk = len(k_grid)
    Omega = np.zeros((nk, nk))
    dk = k_grid[1] - k_grid[0]
    
    for i in range(nk):
        for j in range(nk):
            k = np.array([k_grid[i], k_grid[j]])
            
            # Get eigenvectors from H at neighboring points
            _, v_center = np.linalg.eigh(H_func(k))
            _, v_right = np.linalg.eigh(H_func(k + np.array([dk, 0])))
            _, v_up = np.linalg.eigh(H_func(k + np.array([0, dk])))
            _, v_up_right = np.linalg.eigh(H_func(k + np.array([dk, dk])))
            
            # Compute phase factors (for band 0)
            phase1 = np.exp(1j * np.angle(v_center[0, 0].conj() @ v_right[0, 0]))
            phase2 = np.exp(1j * np.angle(v_right[0, 0].conj() @ v_up_right[0, 0]))
            phase3 = np.exp(1j * np.angle(v_up_right[0, 0].conj() @ v_up[0, 0]))
            phase4 = np.exp(1j * np.angle(v_up[0, 0].conj() @ v_center[0, 0]))
            
            # Wilson loop
            W = phase1 * phase2 * phase3 * phase4
            Omega[i, j] = np.imag(np.log(W)) / (dk**2)
    
    return Omega

def chern_number_2d(Omega: np.ndarray) -> float:
    """Compute Chern number from Berry curvature."""
    return float(np.sum(Omega) / (2 * np.pi))
