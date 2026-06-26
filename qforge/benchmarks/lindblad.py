"""Lindblad solver benchmarks."""
import numpy as np
import time
from typing import List, Dict

def benchmark_lindblad_scaling(dim_list: List[int]) -> Dict[int, Dict]:
    """Benchmark Lindblad integrator."""
    
    results = {}
    
    for dim in dim_list:
        # Random Hamiltonian
        H = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
        H = (H + H.conj().T) / 2
        
        rho0 = np.eye(dim, dtype=complex) / dim
        L = 0.1 * np.random.randn(dim, dim)
        
        times = np.linspace(0, 1, 101)
        
        t0 = time.time()
        from qforge.open_systems.lindblad import LindladSolver
        solver = LindladSolver()
        result = solver.integrate(H, rho0, [L], times)
        elapsed = time.time() - t0
        
        results[dim] = {
            "time": elapsed,
            "n_steps": len(times),
            "time_per_step": elapsed / len(times),
            "memory_mb": dim**2 * 16 / (1024**2),
        }
    
    return results
