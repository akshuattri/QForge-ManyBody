
"""Example 1: Basic Usage - Heisenberg Chain"""

from qforge.models.heisenberg import build_heisenberg_hamiltonian
from qforge.solvers.exact_diag import exact_diagonalization
from qforge.analysis.observables import compute_magnetization

# Parameters
N = 8  # 8 spins
J_x = 1.0
J_z = 1.0
h = 0.5

# Build Hamiltonian
print("Building Heisenberg Hamiltonian (N=8)...")
H = build_heisenberg_hamiltonian(N, J_x=J_x, J_z=J_z, h=h)

# Solve
print("Solving via exact diagonalization...")
result = exact_diagonalization(H, verbose=True)

# Results
E0 = result["eigenvalues"][0]
gap = result["eigenvalues"][1] - result["eigenvalues"][0]
psi0 = result["eigenvectors"][:, 0]

print(f"\nGround state energy: E_0 = {E0:.8f}")
print(f"Energy gap: ΔE = {gap:.8f}")
print(f"Ground state fidelity: {abs(psi0[0]):.6f}")

# Analyze ground state
rho0 = psi0[:, None] @ psi0[None, :].conj()  # Convert to density matrix
M_z = compute_magnetization(rho0, N, direction="z")
print(f"Magnetization: M_z = {M_z:.6f}")
