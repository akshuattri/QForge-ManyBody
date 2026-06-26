
# QForge API Reference

## Core Classes

### ExactDiagonalizationResult
```python
result = exact_diagonalization(H)
result["eigenvalues"]      # np.ndarray of eigenvalues
result["eigenvectors"]     # np.ndarray of eigenvectors
result["computation_time"] # float, seconds
result["converged"]        # bool
```

### Lindblad Result
```python
result = integrate_lindblad(H, rho0, L_ops, times)
result["times"]    # Time points
result["rho_t"]    # Density matrices at each time
```

## Key Functions

### Solvers
- `exact_diagonalization(H, verbose=False)`
- `integrate_lindblad(H, rho0, L_ops, times, rtol=1e-8)`

### Models
- `build_heisenberg_hamiltonian(N, J_x, J_y, J_z, h, bc)`
- `build_ising_hamiltonian(N, J, h, bc)`
- `build_ssh_hamiltonian(N, v, w)`

### Analysis
- `compute_magnetization(rho, N, direction)`
- `compute_entropy(rho)`
- `compute_purity(rho)`
- `compute_correlation(rho, N, i, j, direction)`

### Geometry
- `berry_curvature_2band(k_grid, H_func)`
- `chern_number_2d(Omega)`

## Examples

See `examples/` directory for complete working examples.
