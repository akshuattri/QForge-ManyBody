"""Chern number calculation."""
import numpy as np

def chern_number_from_curvature(Omega: np.ndarray) -> float:
    """Chern number = ∫ Ω dk / 2π."""
    return float(np.sum(Omega) / (2 * np.pi))

def wilson_loop_chern(u_func, kx_grid: np.ndarray, ky_grid: np.ndarray) -> float:
    """Compute Chern number via Wilson loop."""
    nky = len(ky_grid)
    phases = []
    
    for i, kx in enumerate(kx_grid):
        log_W = 0.0
        for j in range(nky):
            ky1 = ky_grid[j]
            ky2 = ky_grid[(j+1) % nky]
            
            u1 = u_func(np.array([kx, ky1]))
            u2 = u_func(np.array([kx, ky2]))
            
            overlap = np.dot(u1.conj(), u2)
            log_W += np.angle(overlap)
        
        phases.append(log_W)
    
    return float(np.mean(np.gradient(phases)) / (2*np.pi))
