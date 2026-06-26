
# QForge Physics Guide

## Hamiltonian Models

### Heisenberg Model

H = Σ_i [J_x S_i^x S_{i+1}^x + J_y S_i^y S_{i+1}^y + J_z S_i^z S_{i+1}^z] + h Σ_i S_i^z

Variants:
- XXX model: J_x = J_y = J_z (isotropic)
- XXZ model: J_x = J_y ≠ J_z (anisotropic)
- Ising limit: J_x = J_y = 0 (classical)

### Transverse-Field Ising Model

H = -Σ_i σ_z^i σ_z^{i+1} - h Σ_i σ_x^i

Shows quantum phase transition at h = 1.

### SSH Model (Topological)

H = Σ_i [v a†_i b_i + w b†_i a_{i+1}] + h.c.

Topological phase when v < w (edge states protected).

## Solvers

### Exact Diagonalization
- Full eigendecomposition
- Suitable for dim ≤ 2^15
- Validation against analytical solutions

### Lindblad Master Equation
- Open quantum system dynamics
- dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k† L_k, ρ})
- Trace and Hermiticity preservation guaranteed

## Observables

- **Magnetization**: M = Σ_i ⟨S_i^z⟩
- **Entanglement Entropy**: S = -Tr(ρ ln ρ)
- **Correlations**: C(i,j) = ⟨S_i S_j⟩
- **Purity**: P = Tr(ρ²)

## Validation Benchmarks

All implementations validated against:
1. Analytical solutions (Bethe ansatz)
2. Published papers
3. QuTiP reference code
4. Conservation law checks
