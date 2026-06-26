# Lindblad Dynamics — Dissipation-Induced Phase Transition

Reproduces the phase-diagram calculation showing how increasing dissipation
strength drives a 2-level driven system through a quantum phase transition,
measured by the von Neumann entropy of the steady state.

## Physics

For a driven-dissipative 2-level system with Hamiltonian coupling `Ω` and
Lindblad decay rate `γ`, the steady state interpolates between:

- **Coherent regime** (`γ ≪ Ω`): near-pure state, low entropy
- **Dissipative regime** (`γ ≫ Ω`): ground state, zero entropy
- **Crossover** (`γ ~ Ω`): entropy peaks as competition is maximal

## Reproduce

```bash
python reproduce.py
```

Expected output:
```
Reproducing Figure 1: Dissipation-Induced Phase Transition
  Phase transition at γ ≈ 1.0
  S(γ=0) = 0.6931   (maximally mixed)
  S(γ=2) = 0.0003   (near-pure ground state)
Reproducing Figure 2: Steady-State Properties
  NESS purity: 0.5000
  NESS entropy: 0.6931
```

## References

```bibtex
@article{Breuer2002,
  author    = {Breuer, H.-P. and Petruccione, F.},
  title     = {The Theory of Open Quantum Systems},
  publisher = {Oxford University Press},
  year      = {2002}
}

@article{Lindblad1976,
  author  = {Lindblad, G.},
  title   = {On the generators of quantum dynamical semigroups},
  journal = {Communications in Mathematical Physics},
  volume  = {48},
  pages   = {119--130},
  year    = {1976}
}
```
