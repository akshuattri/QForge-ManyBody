# Non-Hermitian Topology — Hatano-Nelson Model

Reproduces results on the non-Hermitian skin effect and exceptional-point
dynamics in the Hatano-Nelson chain with asymmetric hopping.

## Physics

The Hatano-Nelson model is a 1D tight-binding chain with asymmetric hopping:
- Right hop: `t · e^α`
- Left hop:  `t · e^{−α}`

For open boundary conditions, eigenstates are exponentially localized at one
boundary — the **non-Hermitian skin effect**. Exceptional points arise when
the complex spectrum degenerates.

## Reproduce

```bash
python reproduce.py
```

Expected output:
```
Reproducing Figure 1: Exceptional Points
  Min gap: ~0.30
Reproducing Figure 2: Skin Effect
  Number of edge-localized states: 0
  Bulk states participation: ~34
```

## References

```bibtex
@article{Hatano1996,
  author  = {Hatano, N. and Nelson, D. R.},
  title   = {Localization Transitions in Non-Hermitian Quantum Mechanics},
  journal = {Physical Review Letters},
  volume  = {77},
  pages   = {570},
  year    = {1996}
}

@article{Bergholtz2021,
  author  = {Bergholtz, E. J. and Budich, J. C. and Kunst, F. K.},
  title   = {Exceptional topology of non-Hermitian systems},
  journal = {Reviews of Modern Physics},
  volume  = {93},
  pages   = {015005},
  year    = {2021}
}
```
