"""Heisenberg model benchmarks."""
import numpy as np
import time
from typing import List, Dict

def benchmark_heisenberg_spectrum(N_list: List[int]) -> Dict[int, Dict]:
    """Benchmark spectrum computation."""
    
    results = {}
    
    for N in N_list:
        from qforge.models.heisenberg import build_heisenberg_hamiltonian
        from qforge.solvers.exact_diag import exact_diagonalization
        
        H = build_heisenberg_hamiltonian(N, J_x=1, J_y=1, J_z=1, h=0)
        
        t0 = time.time()
        result = exact_diagonalization(H)
        elapsed = time.time() - t0
        
        results[N] = {
            "dim": 2**N,
            "time": elapsed,
            "E_0": result["eigenvalues"][0],
            "gap": result["eigenvalues"][1] - result["eigenvalues"][0],
            "memory_mb": (2**N)**2 * 16 / (1024**2),
        }
    
    return results

def benchmark_heisenberg_scaling():
    """Scaling analysis."""
    
    N_values = list(range(6, 17))
    benchmark_heisenberg_spectrum(N_values)
    
    print("N | Dim     | Time (s) | Memory (MB)")
    print("-" * 40)
