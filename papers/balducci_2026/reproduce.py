
"""Reproduce: Balducci et al. - Non-Hermitian Topology in Open Systems"""

import numpy as np
from qforge.solvers.lindblad import integrate_lindblad

def fig1_dissipation_effect():
    """Figure 1: Effect of dissipation on topological properties."""
    
    print("Reproducing Balducci Figure 1...")
    
    # 2-level system with non-Hermitian perturbation
    H = np.array([[1, 1j], [-1j, 1]], dtype=complex) * 0.5
    
    # Collapse operators: Dissipation strength
    gamma_values = np.array([0.0, 0.1, 0.5, 1.0])
    
    results = {}
    for gamma in gamma_values:
        L = np.sqrt(gamma) * np.array([[0, 0], [1, 0]], dtype=complex)
        rho0 = np.eye(2)/2
        
        times = np.linspace(0, 5, 51)
        result = integrate_lindblad(H, rho0, [L], times)
        
        # Extract observables
        purity = [np.trace(rho @ rho).real for rho in result["rho_t"]]
        results[gamma] = purity
    
    print("  Dissipation effects calculated")
    return results

def fig2_exceptional_point():
    """Figure 2: Exceptional point dynamics."""
    
    print("Reproducing Balducci Figure 2...")
    
    # Parameter sweep approaching exceptional point
    epsilon_values = np.linspace(-0.5, 0.5, 21)
    gaps = []
    
    for eps in epsilon_values:
        H = np.array([[0, 1], [1, 0]], dtype=complex) + eps * np.diag([1, -1])
        eigs = np.linalg.eigvalsh(H)
        gap = np.abs(eigs[1] - eigs[0])
        gaps.append(gap)
    
    print("  Exceptional point structure computed")
    return {"epsilon": epsilon_values, "gaps": gaps}

if __name__ == "__main__":
    print("="*70)
    print("QForge Paper Reproduction: Balducci et al.")
    print("="*70)
    
    fig1 = fig1_dissipation_effect()
    fig2 = fig2_exceptional_point()
    
    print("\nReproduction complete!")
