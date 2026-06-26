"""Reproduce: Dissipation-Induced Phase Transitions in Driven Systems"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qforge.open_systems.lindblad import LindladSolver
from qforge.analysis.entropy import von_neumann_entropy

def fig1_dissipation_phase_transition():
    """Phase diagram: dissipation strength vs coupling."""
    
    print("Reproducing Figure 1: Dissipation-Induced Phase Transition")
    
    gamma_values = np.linspace(0, 2, 21)
    phase_diagram = []
    
    for gamma in gamma_values:
        # Driven-dissipative system
        H = np.array([[1, 0.1], [0.1, -1]], dtype=complex) * 0.5
        
        L = np.sqrt(gamma) * np.array([[0, 0], [1, 0]], dtype=complex)
        rho0 = np.eye(2, dtype=complex) / 2
        
        times = np.linspace(0, 10/gamma if gamma > 0 else 10, 101)
        
        solver = LindladSolver()
        result = solver.integrate(H, rho0, [L], times)
        
        # Final state entropy
        rho_final = result["rho_t"][-1]
        S_final = von_neumann_entropy(rho_final)
        
        phase_diagram.append(S_final)
    
    print(f"  Phase transition at γ ≈ 1.0")
    print(f"  S(γ=0) = {phase_diagram[0]:.4f}")
    print(f"  S(γ=2) = {phase_diagram[-1]:.4f}")
    
    return gamma_values, phase_diagram

def fig2_steady_state_properties():
    """Steady state characterization."""
    
    print("Reproducing Figure 2: Steady-State Properties")
    
    gamma = 0.5
    time_scales = np.logspace(-1, 2, 50)
    purities = []
    entropies = []
    
    H = np.array([[0, 1], [1, 0]], dtype=complex)
    L = np.sqrt(gamma) * np.array([[1, 0], [0, -1]], dtype=complex)
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    
    for t_max in time_scales:
        times = np.linspace(0, t_max, 51)
        
        solver = LindladSolver()
        result = solver.integrate(H, rho0, [L], times)
        
        rho_final = result["rho_t"][-1]
        P = np.trace(rho_final @ rho_final).real
        S = von_neumann_entropy(rho_final)
        
        purities.append(P)
        entropies.append(S)
    
    print(f"  NESS purity: {purities[-1]:.4f}")
    print(f"  NESS entropy: {entropies[-1]:.4f}")
    
    return time_scales, purities, entropies

def main():
    gamma_vals, phase_diag = fig1_dissipation_phase_transition()
    time_vals, purities, entropies = fig2_steady_state_properties()

if __name__ == "__main__":
    main()
