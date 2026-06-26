# SSH Topological Model — Edge States

Reproduces the key results of the Su-Schrieffer-Heeger (SSH) model demonstrating
topological edge states and the bulk-boundary correspondence.

## Physics

The SSH chain has two hopping parameters: intra-cell `v` and inter-cell `w`.

- **Topological phase** (`v < w`): two zero-energy edge states, winding number = 1
- **Trivial phase** (`v > w`): gapped bulk, no edge states, winding number = 0
- **Phase boundary** (`v = w`): gap closes, topological transition

## Reproduce

```bash
python reproduce.py
```

Expected output:
```
SSH Topological Model: Edge State Analysis
  Topological phase gap: ~0.0000   (edge states at zero energy)
  Trivial phase gap:     ~1.0000
  Mid-gap energy (topo): ~0.000000 (should be ≈ 0)
  Mid-gap energy (triv): ~0.533327
```

## References

```bibtex
@article{Su1979,
  author  = {Su, W. P. and Schrieffer, J. R. and Heeger, A. J.},
  title   = {Solitons in Polyacetylene},
  journal = {Physical Review Letters},
  volume  = {42},
  pages   = {1698},
  year    = {1979}
}
```
