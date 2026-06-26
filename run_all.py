"""QForge Master Runner — runs all examples and paper reproductions, saves plots."""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure qforge is importable
sys.path.insert(0, os.path.dirname(__file__))

RESULTS = os.path.join(os.path.dirname(__file__), "results", "output")
os.makedirs(RESULTS, exist_ok=True)

def save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────
# 1. HEISENBERG CHAIN — exact diagonalization
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("1. Heisenberg Chain  (N=4,6,8,10,12)")
print("="*60)
from qforge.models.heisenberg import build_heisenberg_hamiltonian
from qforge.solvers.exact_diag import exact_diagonalization
from qforge.analysis.observables import compute_magnetization

sizes   = [4, 6, 8, 10]
E0s, gaps, times_ed = [], [], []
for N in sizes:
    t0 = time.time()
    H  = build_heisenberg_hamiltonian(N, J_x=1.0, J_z=1.0, h=0.5)
    res = exact_diagonalization(H, verbose=False)
    ev  = res["eigenvalues"]
    E0s.append(ev[0] / N)          # energy per site
    gaps.append(ev[1] - ev[0])
    times_ed.append(time.time() - t0)
    rho0 = res["eigenvectors"][:, 0:1] @ res["eigenvectors"][:, 0:1].conj().T
    Mz   = compute_magnetization(rho0, N, direction="z")
    print(f"  N={N:2d}  E0/N={ev[0]/N:.5f}  gap={ev[1]-ev[0]:.5f}  Mz={Mz:.4f}  t={times_ed[-1]:.2f}s")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(sizes, E0s, "o-", color="#2563eb", lw=2, ms=7)
axes[0].set(xlabel="System size N", ylabel="E₀ / N", title="Ground-state energy per site")
axes[0].grid(True, alpha=0.3)

axes[1].plot(sizes, gaps, "s-", color="#dc2626", lw=2, ms=7)
axes[1].set(xlabel="N", ylabel="ΔE", title="Spectral gap")
axes[1].grid(True, alpha=0.3)

axes[2].plot(sizes, times_ed, "^-", color="#16a34a", lw=2, ms=7)
axes[2].set(xlabel="N", ylabel="Wall-time (s)", title="ED computation time")
axes[2].grid(True, alpha=0.3)

fig.suptitle("Heisenberg XXZ Chain  (Jx=1, Jz=1, h=0.5)", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "01_heisenberg.png")

# ─────────────────────────────────────────────
# 2. LINDBLAD OPEN SYSTEMS — decay
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("2. Dissipative 2-Level System (Lindblad)")
print("="*60)
from qforge.solvers.lindblad import integrate_lindblad

H     = np.zeros((2, 2), dtype=complex)
gamma = 1.0
decay = np.sqrt(gamma) * np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1|
rho0  = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
times = np.linspace(0, 6 / gamma, 200)

res   = integrate_lindblad(H, rho0, [decay], times, verbose=True)
Pe    = [res["rho_t"][i, 1, 1].real for i in range(len(times))]
Pg    = [res["rho_t"][i, 0, 0].real for i in range(len(times))]
coh   = [abs(res["rho_t"][i, 0, 1])    for i in range(len(times))]
Pe_an = 0.5 * np.exp(-gamma * times)
err   = np.max(np.abs(np.array(Pe) - Pe_an))
print(f"  Max error vs analytical: {err:.2e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(times * gamma, Pe, "r-",  lw=2, label="|1⟩ (excited)")
ax1.plot(times * gamma, Pg, "b-",  lw=2, label="|0⟩ (ground)")
ax1.plot(times * gamma, Pe_an, "k--", lw=1.5, label="Analytical exp(−γt)/2")
ax1.set(xlabel="γt", ylabel="Population", title="State populations")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(times * gamma, coh, "purple", lw=2)
ax2.set(xlabel="γt", ylabel="|ρ₀₁|", title="Coherence decay")
ax2.grid(True, alpha=0.3)

fig.suptitle("Lindblad Decay: 2-Level System", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "02_lindblad_decay.png")

# ─────────────────────────────────────────────
# 3. SSH TOPOLOGICAL MODEL
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("3. SSH Topological Model")
print("="*60)
from qforge.models.ssh import build_ssh_hamiltonian

# SSH uses params: v=intra-cell, w=inter-cell
# Topological: w > v  (w=strong, v=weak)  →  edge states
# Trivial:     v > w
N_ssh  = 20   # N unit cells → 2N sites
v_vals = np.linspace(0.1, 1.9, 60)
w      = 1.0
mid_gaps = []

for v in v_vals:
    H    = build_ssh_hamiltonian(N_ssh, v=v, w=w)
    eigs = np.linalg.eigvalsh(H)
    mid  = len(eigs) // 2
    mid_gaps.append(eigs[mid] - eigs[mid - 1])

topo_region    = v_vals < w
trivial_region = v_vals >= w
print(f"  Topological phase   gap (avg): {np.mean(np.array(mid_gaps)[topo_region]):.4f}")
print(f"  Trivial phase       gap (avg): {np.mean(np.array(mid_gaps)[trivial_region]):.4f}")

# Edge-state profile
H_topo = build_ssh_hamiltonian(N_ssh, v=0.3, w=1.0)
eigs_t, evecs_t = np.linalg.eigh(H_topo)
edge_lo = evecs_t[:, N_ssh - 1]
edge_hi = evecs_t[:, N_ssh]

H_triv = build_ssh_hamiltonian(N_ssh, v=1.7, w=1.0)
eigs_tr, evecs_tr = np.linalg.eigh(H_triv)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(v_vals, mid_gaps, lw=2, color="#2563eb")
axes[0].axvline(1.0, color="k", ls="--", lw=1, label="v=w (phase boundary)")
axes[0].fill_between(v_vals, mid_gaps, where=topo_region, alpha=0.15, color="green", label="Topological (v<w)")
axes[0].fill_between(v_vals, mid_gaps, where=trivial_region, alpha=0.15, color="red", label="Trivial (v>w)")
axes[0].set(xlabel="v (intra-cell hopping)", ylabel="Mid-gap ΔE", title="Phase diagram")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

n_sites = len(edge_lo)
axes[1].bar(range(n_sites), np.abs(edge_lo)**2, color="#16a34a", alpha=0.8, label="Edge state −")
axes[1].bar(range(n_sites), np.abs(edge_hi)**2, color="#dc2626", alpha=0.5, label="Edge state +")
axes[1].set(xlabel="Site index", ylabel="|ψ|²", title="Topological edge states (v=0.3, w=1.0)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

axes[2].plot(eigs_t,  "o", ms=4, color="#16a34a", label="Topological (v=0.3)")
axes[2].plot(eigs_tr, "s", ms=4, color="#dc2626", label="Trivial (v=1.7)")
axes[2].axhline(0, color="k", lw=0.8, ls="--")
axes[2].set(xlabel="State index", ylabel="Energy", title="Spectrum")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

fig.suptitle("SSH Model — Topological Phase Transition", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "03_ssh_topological.png")

# ─────────────────────────────────────────────
# 4. HATANO-NELSON — non-Hermitian skin effect
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("4. Hatano-Nelson Non-Hermitian Model")
print("="*60)
from qforge.models.hatano_nelson import build_hatano_nelson

N_hn     = 40
alphas   = np.linspace(0.0, 0.5, 30)
mid_gaps_hn = []

for alpha in alphas:
    H    = build_hatano_nelson(N_hn, alpha=alpha)
    eigs = np.linalg.eigvalsh(H)   # imaginary part near zero for real eigenvalues
    mid  = N_hn // 2
    mid_gaps_hn.append(eigs[mid] - eigs[mid - 1])

# Skin effect: participation ratio
alpha_skin = 0.3
H_skin = build_hatano_nelson(N_hn, alpha=alpha_skin)
eigs_s, evecs_s = np.linalg.eigh(H_skin)
PR = [1.0 / np.sum(np.abs(evecs_s[:, i])**4) for i in range(N_hn)]

alpha0 = 0.0
H0 = build_hatano_nelson(N_hn, alpha=alpha0)
eigs0, evecs0 = np.linalg.eigh(H0)

print(f"  Min mid-gap (α sweep): {min(mid_gaps_hn):.5f}")
print(f"  Avg participation ratio (α=0.3): {np.mean(PR):.2f}  (bulk~N/2={N_hn//2})")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(alphas, mid_gaps_hn, "o-", color="#7c3aed", lw=2, ms=5)
axes[0].set(xlabel="Asymmetry α", ylabel="Mid-gap ΔE", title="Gap vs asymmetry")
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(N_hn), PR, color="#7c3aed", alpha=0.8)
axes[1].axhline(N_hn / 4, color="k", ls="--", lw=1, label=f"N/4={N_hn//4}")
axes[1].set(xlabel="Eigenstate index", ylabel="Participation ratio", title=f"Skin effect (α={alpha_skin})")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(eigs0, "o", ms=4, color="gray", label="α=0 (Hermitian)")
axes[2].plot(eigs_s, "s", ms=4, color="#7c3aed", label=f"α={alpha_skin}")
axes[2].set(xlabel="State index", ylabel="Energy", title="Spectrum")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

fig.suptitle("Hatano-Nelson: Non-Hermitian Skin Effect", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "04_hatano_nelson.png")

# ─────────────────────────────────────────────
# 5. BALDUCCI 2026 — dissipation in topology
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("5. Balducci et al. 2026 — Dissipation & topology")
print("="*60)
from qforge.solvers.lindblad import integrate_lindblad as ilb

H2 = np.array([[1, 1j], [-1j, 1]], dtype=complex) * 0.5
gamma_vals = [0.0, 0.1, 0.5, 1.0, 2.0]
times2 = np.linspace(0, 8, 120)
rho0_2 = np.eye(2, dtype=complex) / 2
colors = ["#111827", "#1d4ed8", "#16a34a", "#dc2626", "#9333ea"]

purity_curves = {}
for gv in gamma_vals:
    L   = np.sqrt(gv) * np.array([[0, 1], [0, 0]], dtype=complex)
    res = ilb(H2, rho0_2, [L], times2)
    pur = [np.trace(res["rho_t"][i] @ res["rho_t"][i]).real for i in range(len(times2))]
    purity_curves[gv] = pur

# EP sweep
eps_vals = np.linspace(-0.5, 0.5, 80)
ep_gaps  = []
for eps in eps_vals:
    H_ep = np.array([[0, 1], [1, 0]], dtype=complex) + eps * np.diag([1.0, -1.0])
    ev   = np.linalg.eigvalsh(H_ep)
    ep_gaps.append(abs(ev[1] - ev[0]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for gv, col in zip(gamma_vals, colors):
    ax1.plot(times2, purity_curves[gv], lw=2, color=col, label=f"γ={gv}")
ax1.set(xlabel="Time", ylabel="Purity Tr(ρ²)", title="Dissipation-driven purity decay")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(eps_vals, ep_gaps, lw=2, color="#dc2626")
ax2.set(xlabel="ε (detuning)", ylabel="Gap |E₁−E₀|", title="Exceptional point (gap closes at ε=0)")
ax2.axvline(0, color="k", ls="--", lw=1)
ax2.grid(True, alpha=0.3)

fig.suptitle("Balducci et al. 2026 — Open System Topology", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "05_balducci_2026.png")

# ─────────────────────────────────────────────
# 6. LINDBLAD PHASE TRANSITION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("6. Lindblad Dynamics — Dissipation Phase Transition")
print("="*60)
from qforge.open_systems.lindblad import LindladSolver

solver  = LindladSolver()
H3      = np.array([[1, 0], [0, -1]], dtype=complex) * 0.5
times3  = np.linspace(0, 10, 150)
rho0_3  = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
grange  = np.linspace(0, 3, 25)

entropy_ss, purity_ss = [], []
for gv in grange:
    L   = np.sqrt(gv) * np.array([[0, 1], [0, 0]], dtype=complex)
    res = solver.integrate(H3, rho0_3, [L], times3)
    rho_ss = res["rho_t"][-1]
    ev2    = np.linalg.eigvalsh(rho_ss)
    ev2    = np.clip(ev2, 1e-12, None)
    S      = -np.sum(ev2 * np.log(ev2))
    P      = np.trace(rho_ss @ rho_ss).real
    entropy_ss.append(S)
    purity_ss.append(P)

print(f"  Entropy at γ=0  : {entropy_ss[0]:.4f}  (max-mixed = ln2 ≈ 0.693)")
print(f"  Entropy at γ=3  : {entropy_ss[-1]:.4f}  (pure state → 0)")
print(f"  Purity  at γ=0  : {purity_ss[0]:.4f}")
print(f"  Purity  at γ=3  : {purity_ss[-1]:.4f}")

# Time traces for a few γ values
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for gv, col in zip([0.0, 0.5, 1.0, 2.0], ["gray", "#1d4ed8", "#16a34a", "#dc2626"]):
    L   = np.sqrt(gv) * np.array([[0, 1], [0, 0]], dtype=complex)
    res = solver.integrate(H3, rho0_3, [L], times3)
    Pe_t = [res["rho_t"][i, 1, 1].real for i in range(len(times3))]
    axes[0].plot(times3, Pe_t, lw=2, color=col, label=f"γ={gv}")
axes[0].set(xlabel="Time", ylabel="ρ₁₁ (excited pop.)", title="Time evolution")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].plot(grange, entropy_ss, "o-", color="#7c3aed", lw=2, ms=5)
axes[1].axhline(np.log(2), color="k", ls="--", lw=1, label="ln 2 (max-mixed)")
axes[1].set(xlabel="Dissipation γ", ylabel="von Neumann entropy S", title="Steady-state entropy")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(grange, purity_ss, "s-", color="#dc2626", lw=2, ms=5)
axes[2].set(xlabel="γ", ylabel="Tr(ρ²)", title="Steady-state purity")
axes[2].grid(True, alpha=0.3)

fig.suptitle("Lindblad Phase Transition: Entropy vs Dissipation", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "06_lindblad_phase.png")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ALL DONE — Results written to:")
print(f"  {RESULTS}")
imgs = [f for f in os.listdir(RESULTS) if f.endswith(".png")]
for img in sorted(imgs):
    print(f"  {img}")
print("="*60)
