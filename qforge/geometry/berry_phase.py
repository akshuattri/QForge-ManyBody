"""Berry phase and Berry curvature."""
import numpy as np
from typing import Callable, Tuple

def berry_phase_1d(k_path: np.ndarray, H_func: Callable,
                   band: int = 0) -> float:
    """Compute Berry phase along 1D path."""
    
    nk = len(k_path)
    phase = 0.0
    
    for i in range(nk-1):
        k1 = k_path[i]
        k2 = k_path[i+1]
        
        _, u1 = np.linalg.eigh(H_func(k1))
        _, u2 = np.linalg.eigh(H_func(k2))
        
        u1_band = u1[:, band]
        u2_band = u2[:, band]
        
        overlap = np.dot(u1_band.conj(), u2_band)
        phase += np.angle(overlap)
    
    return phase

def berry_curvature_2d(kx_grid: np.ndarray, ky_grid: np.ndarray,
                       H_func: Callable, band: int = 0) -> np.ndarray:
    """Compute Berry curvature in 2D."""
    
    nkx, nky = len(kx_grid), len(ky_grid)
    Omega = np.zeros((nkx, nky))
    
    dkx = kx_grid[1] - kx_grid[0] if len(kx_grid) > 1 else 1.0
    dky = ky_grid[1] - ky_grid[0] if len(ky_grid) > 1 else 1.0
    
    for i in range(nkx-1):
        for j in range(nky-1):
            k00 = np.array([kx_grid[i], ky_grid[j]])
            k10 = np.array([kx_grid[i+1], ky_grid[j]])
            k01 = np.array([kx_grid[i], ky_grid[j+1]])
            k11 = np.array([kx_grid[i+1], ky_grid[j+1]])
            
            _, u00 = np.linalg.eigh(H_func(k00))
            _, u10 = np.linalg.eigh(H_func(k10))
            _, u01 = np.linalg.eigh(H_func(k01))
            _, u11 = np.linalg.eigh(H_func(k11))
            
            u00_band = u00[:, band]
            u10_band = u10[:, band]
            u01_band = u01[:, band]
            u11_band = u11[:, band]
            
            # Plaquette Berry phase
            phase = (np.angle(np.dot(u00_band.conj(), u10_band)) +
                    np.angle(np.dot(u10_band.conj(), u11_band)) +
                    np.angle(np.dot(u11_band.conj(), u01_band)) +
                    np.angle(np.dot(u01_band.conj(), u00_band)))
            
            Omega[i, j] = phase / (dkx * dky)
    
    return Omega
