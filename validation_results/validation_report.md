# QForge Validation Report

Generated: 2026-06-26 08:33 UTC

**14/14 checks passed**

## Results

| Check | Status | Expected | Computed | Rel. Error |
|---|---|---|---|---|
| 2-spin dimer  E₀ | ✅ PASS | `-3` | `-3` | `0.00e+00` |
| 2-spin dimer  gap (4J) | ✅ PASS | `4` | `4` | `0.00e+00` |
| 4-spin chain  E₀ | ✅ PASS | `-6.4641` | `-6.4641` | `0.00e+00` |
| Hilbert-space dim (N=6) | ✅ PASS | `64` | `64` | `0.00e+00` |
| Lindblad decay  max |P_e − analytical| | ✅ PASS | `0` | `4.20039e-07` | `4.20e+23` |
| Trace preservation  max |Tr(ρ) − 1| | ✅ PASS | `0` | `2.22045e-16` | `2.22e+14` |
| Hermiticity  max |ρ − ρ†| | ✅ PASS | `0` | `0` | `0.00e+00` |
| Steady state  ρ₀₀ → 1 | ✅ PASS | `1` | `1` | `2.82e-09` |
| Topological edge state  |E_edge| | ✅ PASS | `0` | `3.17296e-11` | `3.17e+19` |
| Trivial phase gap > 0.5 | ✅ PASS | `1.44696` | `1.44696` | `0.00e+00` |
| Phase boundary (v=w)  gap < trivial (finite-size scaling) | ✅ PASS | `1.44696` | `0.153211` | `8.94e-01` |
| Ground-state magnetization Mz (h=0) | ✅ PASS | `0` | `2.77556e-17` | `2.78e+13` |
| Max-mixed state entropy  S = ln 2 | ✅ PASS | `0.693147` | `0.693147` | `0.00e+00` |
| Pure state entropy  S = 0 | ✅ PASS | `0` | `-0` | `0.00e+00` |

## Sections

- **A. Exact Diagonalization** — Heisenberg dimer & chain vs analytical
- **B. Lindblad Dynamics** — decay, trace, Hermiticity, steady state
- **C. SSH Topological Model** — edge states, trivial gap, phase boundary
- **D. Physical Observables** — magnetization, von Neumann entropy
