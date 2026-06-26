# QForge Tutorial: Getting Started

## 1. Exact Diagonalization

```python
import numpy as np
from qforge.models.heisenberg import build_heisenberg_hamiltonian
from qforge.solvers.exact_diag import exact_diagonalization

# Build 8-spin Heisenberg XXZ
H = build_heisenberg_hamiltonian(
    N=8,
    J_x=1.0, J_y=1.0, J_z=1.0,  # Coupling constants
    h=0.5,  # Magnetic field
    bc="open"  # Open boundary condition
)

# Solve exactly
result = exact_diagonalization(H, verbose=True)

# Results
E_0 = result["eigenvalues"][0]  # Ground state energy
gap = result["eigenvalues"][1] - result["eigenvalues"][0]
psi_0 = result["eigenvectors"][:, 0]  # Ground state

print(f"E_0 = {E_0:.8f}")
print(f"Gap = {gap:.8f}")
```

## 2. Open Quantum Systems (Lindblad)

```python
from qforge.open_systems.lindblad import LindladSolver
from qforge.analysis.entropy import von_neumann_entropy

# 2-level system with decay
H = np.zeros((2, 2), dtype=complex)
decay_op = np.array([[0, 0], [1, 0]], dtype=complex)

# Initial state: superposition
rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

# Time evolution
times = np.linspace(0, 5, 101)
solver = LindladSolver()
result = solver.integrate(H, rho0, [decay_op], times)

# Analyze
for i in [0, 50, 100]:
    rho = result["rho_t"][i]
    S = von_neumann_entropy(rho)
    P = np.trace(rho @ rho).real
    print(f"t={times[i]:.1f}: S={S:.4f}, P={P:.4f}")
```

## 3. Berry Phase & Topology

```python
from qforge.geometry.berry_phase import berry_phase_1d, berry_curvature_2d
from qforge.models.ssh import build_ssh_hamiltonian

# SSH model along parameter path
k_values = np.linspace(0, 2*np.pi, 101)

def H_func(k):
    v = 0.8 + 0.3 * np.cos(k)
    w = 1.0 + 0.2 * np.sin(k)
    # ... build SSH with these parameters
    return build_ssh_hamiltonian(10, v, w)

# Compute Berry phase
gamma = berry_phase_1d(k_values, H_func, band=0)
print(f"Berry phase: {gamma:.6f}")
```

## 4. Validation & Benchmarks

```python
# Run validation suite
import subprocess
subprocess.run(["pytest", "validation/", "-v"])

# Run benchmarks
from qforge.benchmarks.heisenberg import benchmark_heisenberg_spectrum

N_list = [6, 8, 10, 12, 14]
results = benchmark_heisenberg_spectrum(N_list)

for N, data in results.items():
    print(f"N={N}: dim={data['dim']}, time={data['time']:.3f}s")
```

## Next Steps
- Explore `examples/` for working code
- Check `papers/` for research reproductions
- Read `docs/` for API reference
