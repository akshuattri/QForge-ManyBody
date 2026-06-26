#!/usr/bin/env python3
"""
QForge Scientific Benchmark
============================
Validates computed results against known analytical solutions and
published reference values.  Run from the repo root:

    python scripts/benchmark.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qforge.models.heisenberg import build_heisenberg_hamiltonian
from qforge.solvers.exact_diag import exact_diagonalization
from qforge.solvers.lindblad import integrate_lindblad
from qforge.models.ssh import build_ssh_hamiltonian

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
SEP  = "─" * 62

def _result(label, expected, computed, tol=1e-8):
    err = abs(computed - expected) / (abs(expected) + 1e-30)
    ok  = err < tol
    tag = PASS if ok else FAIL
    print(f"    Expected   {expected:.10f}")
    print(f"    Computed   {computed:.10f}")
    print(f"    Rel. error {err:.2e}{tag}")
    return ok


# ──────────────────────────────────────────────
# 1.  Heisenberg dimer  (N=2, exact solution)
# ──────────────────────────────────────────────
print(f"\n{SEP}")
print("1. Heisenberg Dimer  (N=2, J=1, h=0)")
print(SEP)
# Exact: singlet E = -3/4, triplet E = +1/4  (per-spin in units of J)
print("  Ground-state energy E₀")
H2 = build_heisenberg_hamiltonian(2, J_x=1.0, J_z=1.0, h=0.0)
r2 = exact_diagonalization(H2, verbose=False)
E0_2 = r2["eigenvalues"][0]
# Pauli-matrix convention: H = J Σ σᵢ·σⱼ  →  singlet E = -3J, triplet E = +J
ok1  = _result("E₀ = -3J = -3.0  (Pauli convention, singlet)", -3.0, E0_2)

print("\n  Spectral gap ΔE  (triplet − singlet = 4J)")
gap2 = r2["eigenvalues"][1] - r2["eigenvalues"][0]
ok2  = _result("ΔE = 4J = 4.0", 4.0, gap2)

# ──────────────────────────────────────────────
# 2.  Heisenberg 4-site ring  (known ground state)
# ──────────────────────────────────────────────
print(f"\n{SEP}")
print("2. Heisenberg Chain  (N=4, J=1, h=0)")
print(SEP)
# Exact value for N=4 OBC, J=1 Pauli convention verified by independent diagonalization
# E₀/N = −1.616025... → E₀_total = −6.464101...  (4 sites)
E0_ref_4 = -6.464101615138  # verified by exact diagonalization
H4 = build_heisenberg_hamiltonian(4, J_x=1.0, J_z=1.0, h=0.0)
r4 = exact_diagonalization(H4, verbose=False)
E0_4 = r4["eigenvalues"][0]
print("  Ground-state energy E₀  (exact diagonalization, Pauli convention)")
ok3  = _result("E₀ (N=4 OBC reference)", E0_ref_4, E0_4, tol=1e-8)

# ──────────────────────────────────────────────
# 3.  Lindblad decay — analytical solution
# ──────────────────────────────────────────────
print(f"\n{SEP}")
print("3. Lindblad Decay  (2-level, analytical)")
print(SEP)
gamma = 1.0
H0    = np.zeros((2, 2), dtype=complex)
L     = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=complex)
rho0  = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
tmax  = 5.0
times = np.linspace(0, tmax, 500)

t0  = time.time()
res = integrate_lindblad(H0, rho0, [L], times)
dt  = time.time() - t0

Pe_num = np.array([res["rho_t"][i, 1, 1].real for i in range(len(times))])
Pe_ana = 0.5 * np.exp(-gamma * times)
max_err = np.max(np.abs(Pe_num - Pe_ana))
ok4 = max_err < 1e-5
tag = PASS if ok4 else FAIL
print(f"    Max |numerical − analytical|  {max_err:.2e}{tag}")
print(f"    Integration time              {dt*1000:.1f} ms")

# ──────────────────────────────────────────────
# 4.  SSH edge states — topological gap
# ──────────────────────────────────────────────
print(f"\n{SEP}")
print("4. SSH Model  (N=20 cells, v=0.3, w=1.0)")
print(SEP)
H_ssh = build_ssh_hamiltonian(20, v=0.3, w=1.0)
eigs  = np.linalg.eigvalsh(H_ssh)
mid   = len(eigs) // 2
topo_gap = eigs[mid] - eigs[mid - 1]
edge_gap = abs(eigs[mid - 1])   # should be ≈ 0 for topological edge states
ok5 = edge_gap < 0.05
tag = PASS if ok5 else FAIL
print(f"    Mid-gap energy (edge state)   {edge_gap:.6f}  (expect ≈ 0){tag}")
print(f"    Full mid-gap ΔE               {topo_gap:.6f}")

# ──────────────────────────────────────────────
# 5.  Scaling benchmark
# ──────────────────────────────────────────────
print(f"\n{SEP}")
print("5. Exact Diagonalization Scaling")
print(SEP)
print(f"  {'N':>4}  {'Hilbert dim':>12}  {'E₀/N':>12}  {'Gap':>10}  {'Time (s)':>10}")
print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
for N in [4, 6, 8, 10]:
    H  = build_heisenberg_hamiltonian(N, J_x=1.0, J_z=1.0, h=0.5)
    t0 = time.time()
    r  = exact_diagonalization(H, verbose=False)
    elapsed = time.time() - t0
    ev = r["eigenvalues"]
    print(f"  {N:>4}  {2**N:>12d}  {ev[0]/N:>12.6f}  {ev[1]-ev[0]:>10.6f}  {elapsed:>10.3f}")

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
all_pass = all([ok1, ok2, ok3, ok4, ok5])
print(f"\n{SEP}")
if all_pass:
    print("\033[92m  All benchmarks PASSED\033[0m")
else:
    print("\033[91m  Some benchmarks FAILED — see above\033[0m")
print(SEP + "\n")
sys.exit(0 if all_pass else 1)
