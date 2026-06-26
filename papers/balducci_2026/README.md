# Balducci et al. (2026) — Open System Topology

Reproduces key figures from Balducci et al. studying topological properties
in open quantum systems described by Lindblad master equations.

## Physics

The paper investigates how dissipation modifies topological invariants in a
2-level open system. Key results:

- Dissipation-driven purity decay: stronger `γ` → faster decoherence
- Exceptional point at `ε = 0`: energy gap closes under tuning of detuning
- Non-trivial interplay between coherent topology and Lindblad dissipation

## Reproduce

```bash
python reproduce.py
```

Expected output:
```
QForge Paper Reproduction: Balducci et al.
Reproducing Balducci Figure 1...
  Dissipation effects calculated
Reproducing Balducci Figure 2...
  Exceptional point structure computed
Reproduction complete!
```

## References

```bibtex
@article{Balducci2026,
  author  = {Balducci, F. and others},
  title   = {Topological properties of open quantum systems},
  journal = {Physical Review B},
  year    = {2026},
  note    = {Preprint}
}
```
