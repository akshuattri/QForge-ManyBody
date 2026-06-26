
"""Example 2: Open Quantum Systems - Lindblad Dynamics"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qforge.solvers.lindblad import integrate_lindblad

print("Open Quantum Systems Example: Dissipative 2-Level System")
print("="*70)

# Simple decay: H = 0, L = √γ |0⟩⟨1|
H = np.zeros((2, 2), dtype=complex)
gamma = 1.0
decay = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=complex)

# Initial state: superposition
rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

# Time evolution
times = np.linspace(0, 5/gamma, 101)
print(f"Integrating Lindblad equation (0 to {times[-1]:.1f}/γ)...")
result = integrate_lindblad(H, rho0, [decay], times, verbose=True)

# Extract populations
P_e = [result["rho_t"][i, 1, 1].real for i in range(len(times))]
P_g = [result["rho_t"][i, 0, 0].real for i in range(len(times))]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(times*gamma, P_e, 'r-', label='|1⟩ (excited)', linewidth=2)
plt.plot(times*gamma, P_g, 'b-', label='|0⟩ (ground)', linewidth=2)
plt.axhline(1.0, color='b', linestyle='--', alpha=0.3)
plt.xlabel('Time (γt)', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.title('Dissipative Dynamics: Decay to Ground State', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lindblad_decay.png', dpi=150)
print("\nPlot saved to: lindblad_decay.png")

# Analytical verification
P_e_analytical = 0.5 * np.exp(-gamma * times)
error = np.abs(P_e - P_e_analytical)
print(f"Max error vs analytical: {np.max(error):.2e}")
