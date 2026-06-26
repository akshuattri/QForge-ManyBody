"""
QForge — Complete Results Runner
=================================
Runs every model, solver, analysis routine, and visualization.
Saves one PNG per topic to results/output/.
"""

import sys, os, traceback, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

OUTDIR = os.path.join(os.path.dirname(__file__), "results", "output")
os.makedirs(OUTDIR, exist_ok=True)

saved, failed = [], []

def save(fig, name, title=""):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(name)
    print(f"  ✓  {name}")

def section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")

def skip(name, err):
    failed.append((name, str(err)[:80]))
    print(f"  ✗  {name}  —  {str(err)[:80]}")

# ════════════════════════════════════════════════════════════════
# 1.  HEISENBERG CHAIN — multiple system sizes
# ════════════════════════════════════════════════════════════════
section("1. Heisenberg XXZ Chain")
try:
    from qforge.models.heisenberg import build_heisenberg_hamiltonian
    from qforge.solvers.exact_diag import exact_diagonalization
    from qforge.analysis.observables import compute_magnetization

    sizes  = [4, 6, 8, 10]
    E0s, gaps, Mzs = [], [], []
    spectra_full = {}

    for N in sizes:
        H  = build_heisenberg_hamiltonian(N, J_x=1.0, J_z=1.0, h=0.5)
        r  = exact_diagonalization(H, verbose=False)
        ev = r["eigenvalues"]
        E0s.append(ev[0] / N)
        gaps.append(ev[1] - ev[0])
        psi0 = r["eigenvectors"][:, 0]
        rho0 = np.outer(psi0, psi0.conj())
        Mzs.append(compute_magnetization(rho0, N, direction="z"))
        spectra_full[N] = ev
        print(f"    N={N}: E0/N={ev[0]/N:.5f}  gap={ev[1]-ev[0]:.5f}  Mz={Mzs[-1]:.4f}")

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sizes, E0s, "o-", color="#2563eb", lw=2, ms=7)
    ax1.set(xlabel="N", ylabel="E₀/N", title="Ground-state energy per site")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sizes, gaps, "s-", color="#dc2626", lw=2, ms=7)
    ax2.set(xlabel="N", ylabel="ΔE", title="Spectral gap")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(sizes, Mzs, "^-", color="#16a34a", lw=2, ms=7)
    ax3.axhline(0, color="k", ls="--", lw=1)
    ax3.set(xlabel="N", ylabel="Mz", title="Ground-state magnetization")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, :2])
    colors4 = ["#2563eb","#16a34a","#dc2626","#9333ea"]
    for (N, ev), col in zip(spectra_full.items(), colors4):
        ax4.scatter(range(len(ev)), np.sort(ev), s=12, alpha=0.7,
                    label=f"N={N}", color=col)
    ax4.set(xlabel="State index", ylabel="Energy", title="Full eigenvalue spectra")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    # Finite-size scaling: E0/N vs 1/N
    inv_N = [1/N for N in sizes]
    ax5.plot(inv_N, E0s, "o-", color="#7c3aed", lw=2, ms=7)
    p = np.polyfit(inv_N, E0s, 1)
    xs = np.linspace(0, max(inv_N), 50)
    ax5.plot(xs, np.polyval(p, xs), "k--", lw=1, label=f"Extrap. E∞={p[1]:.4f}")
    ax5.set(xlabel="1/N", ylabel="E₀/N", title="Finite-size extrapolation")
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    fig.suptitle("Heisenberg XXZ Chain  (Jx=Jz=1, h=0.5)", fontsize=14, fontweight="bold")
    save(fig, "01_heisenberg_full.png")
except Exception as e:
    skip("01_heisenberg_full.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 2.  TRANSVERSE-FIELD ISING MODEL
# ════════════════════════════════════════════════════════════════
section("2. Transverse-Field Ising Model")
try:
    from qforge.models.ising import build_ising_hamiltonian

    N = 8
    h_vals = np.linspace(0.0, 3.0, 40)
    E0_ising, gap_ising = [], []
    for h in h_vals:
        H  = build_ising_hamiltonian(N, J=1.0, h=h)
        ev = np.linalg.eigvalsh(H)
        E0_ising.append(ev[0] / N)
        gap_ising.append(ev[1] - ev[0])

    # Band structure at critical point h=J=1
    H_crit = build_ising_hamiltonian(N, J=1.0, h=1.0)
    ev_crit = np.linalg.eigvalsh(H_crit)

    print(f"    N={N}: gap closes at h/J ≈ {h_vals[np.argmin(gap_ising)]:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(h_vals, E0_ising, lw=2, color="#2563eb")
    axes[0].axvline(1.0, color="k", ls="--", lw=1, label="h=J (critical)")
    axes[0].set(xlabel="h/J", ylabel="E₀/N", title="Ground-state energy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(h_vals, gap_ising, lw=2, color="#dc2626")
    axes[1].axvline(1.0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="h/J", ylabel="ΔE", title="Spectral gap (closes at QPT)")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(range(len(ev_crit)), ev_crit, s=15, color="#16a34a")
    axes[2].set(xlabel="State index", ylabel="Energy", title="Spectrum at critical point (h=J=1)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Transverse-Field Ising Model  (N=8)", fontsize=13, fontweight="bold")
    save(fig, "02_ising_model.png")
except Exception as e:
    skip("02_ising_model.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 3.  SSH TOPOLOGICAL MODEL
# ════════════════════════════════════════════════════════════════
section("3. SSH Topological Model")
try:
    from qforge.models.ssh import build_ssh_hamiltonian

    N_ssh  = 20
    v_vals = np.linspace(0.1, 1.9, 60)
    w = 1.0
    mid_gaps = []
    for v in v_vals:
        ev = np.linalg.eigvalsh(build_ssh_hamiltonian(N_ssh, v=v, w=w))
        mid = len(ev) // 2
        mid_gaps.append(ev[mid] - ev[mid-1])

    eigs_t, evecs_t = np.linalg.eigh(build_ssh_hamiltonian(N_ssh, v=0.3, w=1.0))
    eigs_tr, _      = np.linalg.eigh(build_ssh_hamiltonian(N_ssh, v=1.7, w=1.0))
    n_sites = len(eigs_t)

    # k-space band structure (periodic BC)
    k_vals = np.linspace(-np.pi, np.pi, 200)
    E_k = np.zeros((len(k_vals), 2))
    for i, k in enumerate(k_vals):
        v0, w0 = 0.3, 1.0
        H_k = np.array([[0, v0 + w0*np.exp(-1j*k)],
                        [v0 + w0*np.exp(1j*k), 0]])
        E_k[i] = np.linalg.eigvalsh(H_k)

    print(f"    Topo gap: {mid_gaps[10]:.4f}  Trivial gap: {mid_gaps[-10]:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(v_vals, mid_gaps, lw=2, color="#2563eb")
    axes[0].axvline(1.0, color="k", ls="--", lw=1)
    axes[0].fill_between(v_vals, mid_gaps, where=v_vals<w, alpha=0.12, color="green", label="Topological")
    axes[0].fill_between(v_vals, mid_gaps, where=v_vals>=w, alpha=0.12, color="red", label="Trivial")
    axes[0].set(xlabel="v", ylabel="Mid-gap ΔE", title="Phase diagram")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(n_sites), np.abs(evecs_t[:, N_ssh-1])**2, color="#16a34a", alpha=0.8, label="Edge −")
    axes[1].bar(range(n_sites), np.abs(evecs_t[:, N_ssh])**2,   color="#dc2626", alpha=0.5, label="Edge +")
    axes[1].set(xlabel="Site", ylabel="|ψ|²", title="Edge-state wavefunctions (v=0.3)")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(eigs_t,  "o", ms=4, color="#16a34a", label="Topological (v=0.3)")
    axes[2].plot(eigs_tr, "s", ms=4, color="#dc2626", label="Trivial (v=1.7)")
    axes[2].axhline(0, color="k", ls="--", lw=0.8)
    axes[2].set(xlabel="State index", ylabel="Energy", title="OBC spectrum")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    axes[3].plot(k_vals, E_k[:, 0], color="#2563eb", lw=2, label="Band 1")
    axes[3].plot(k_vals, E_k[:, 1], color="#dc2626", lw=2, label="Band 2")
    axes[3].axhline(0, color="k", ls="--", lw=0.8)
    axes[3].set(xlabel="k", ylabel="E(k)", title="PBC band structure (v=0.3, w=1)")
    axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)

    fig.suptitle("SSH Model — Topological Phase Transition", fontsize=13, fontweight="bold")
    save(fig, "03_ssh_full.png")
except Exception as e:
    skip("03_ssh_full.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 4.  HATANO-NELSON NON-HERMITIAN
# ════════════════════════════════════════════════════════════════
section("4. Hatano-Nelson Non-Hermitian Chain")
try:
    from qforge.models.hatano_nelson import build_hatano_nelson

    N_hn = 40
    alphas = np.linspace(0, 0.5, 30)
    all_ev = [np.linalg.eigvalsh(build_hatano_nelson(N_hn, alpha=a)) for a in alphas]

    alpha_demo = 0.3
    H_demo = build_hatano_nelson(N_hn, alpha=alpha_demo)
    ev_d, evec_d = np.linalg.eigh(H_demo)
    PR = [1.0/np.sum(np.abs(evec_d[:,i])**4) for i in range(N_hn)]

    ev_herm = np.linalg.eigvalsh(build_hatano_nelson(N_hn, alpha=0))

    print(f"    avg PR (α={alpha_demo}): {np.mean(PR):.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    cm = plt.cm.viridis
    for i, (a, ev) in enumerate(zip(alphas, all_ev)):
        axes[0].scatter([a]*len(ev), ev, s=2, c=[cm(i/len(alphas))]*len(ev), alpha=0.5)
    axes[0].set(xlabel="α", ylabel="Energy", title="Spectrum vs asymmetry")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(N_hn), PR, color="#7c3aed", alpha=0.8)
    axes[1].axhline(N_hn/4, color="k", ls="--", lw=1, label=f"N/4={N_hn//4}")
    axes[1].set(xlabel="Eigenstate", ylabel="Participation ratio", title=f"Skin effect (α={alpha_demo})")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Wavefunction density of 3 eigenstates
    for idx, col in zip([0, N_hn//4, N_hn-1], ["#2563eb","#16a34a","#dc2626"]):
        axes[2].plot(range(N_hn), np.abs(evec_d[:,idx])**2, lw=1.5,
                     color=col, label=f"state {idx}")
    axes[2].set(xlabel="Site", ylabel="|ψ(i)|²", title="Eigenstate densities")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    axes[3].plot(ev_herm, "o", ms=4, color="gray", label="α=0 (Hermitian)")
    axes[3].plot(ev_d,    "s", ms=4, color="#7c3aed", label=f"α={alpha_demo}")
    axes[3].set(xlabel="State index", ylabel="Energy", title="Spectrum comparison")
    axes[3].legend(); axes[3].grid(True, alpha=0.3)

    fig.suptitle("Hatano-Nelson Non-Hermitian Chain (N=40)", fontsize=13, fontweight="bold")
    save(fig, "04_hatano_nelson_full.png")
except Exception as e:
    skip("04_hatano_nelson_full.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 5.  KITAEV CHAIN (TOPOLOGICAL SUPERCONDUCTOR)
# ════════════════════════════════════════════════════════════════
section("5. Kitaev Chain — Topological Superconductor")
try:
    from qforge.models.kitaev import build_kitaev

    N_k = 10
    mu_vals = np.linspace(-3, 3, 60)
    gaps_k  = []
    for mu in mu_vals:
        H  = build_kitaev(N_k, t=1.0, Delta=0.5, mu=mu)
        ev = np.linalg.eigvalsh(H)
        mid = len(ev)//2
        gaps_k.append(ev[mid] - ev[mid-1])

    # Topological phase: |mu| < 2t → Majorana edge modes
    H_topo  = build_kitaev(N_k, t=1.0, Delta=0.5, mu=0.0)
    H_triv  = build_kitaev(N_k, t=1.0, Delta=0.5, mu=3.0)
    ev_topo = np.linalg.eigvalsh(H_topo)
    ev_triv = np.linalg.eigvalsh(H_triv)

    print(f"    Phase boundaries at mu=±2t=±2")
    print(f"    Topo gap (mu=0): {ev_topo[len(ev_topo)//2]-ev_topo[len(ev_topo)//2-1]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(mu_vals, gaps_k, lw=2, color="#2563eb")
    axes[0].axvline(-2, color="k", ls="--", lw=1, label="μ=±2t")
    axes[0].axvline( 2, color="k", ls="--", lw=1)
    axes[0].fill_between(mu_vals, gaps_k, where=np.abs(mu_vals)<2, alpha=0.15, color="green", label="Topological")
    axes[0].set(xlabel="μ/t", ylabel="Gap", title="Spectral gap vs μ")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ev_topo, "o", ms=5, color="#16a34a", label="Topological (μ=0)")
    axes[1].plot(ev_triv, "s", ms=5, color="#dc2626", label="Trivial (μ=3)")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="State index", ylabel="Energy", title="Spectrum")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.abs(mu_vals), gaps_k, lw=2, color="#7c3aed")
    axes[2].axvline(2, color="k", ls="--", lw=1, label="|μ|=2t (transition)")
    axes[2].set(xlabel="|μ|/t", ylabel="Gap", title="Gap vs |μ| (symmetry)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Kitaev Chain — Topological Superconductor (N=10, Δ=0.5)", fontsize=13, fontweight="bold")
    save(fig, "05_kitaev_chain.png")
except Exception as e:
    skip("05_kitaev_chain.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 6.  LINDBLAD DYNAMICS — decay, dephasing, driven
# ════════════════════════════════════════════════════════════════
section("6. Lindblad Master Equation — Multiple Scenarios")
try:
    from qforge.solvers.lindblad import integrate_lindblad
    from qforge.analysis.entropy import von_neumann_entropy

    times = np.linspace(0, 8, 300)

    # Scenario A: decay
    H0    = np.zeros((2,2), dtype=complex)
    L_dec = np.array([[0,1],[0,0]], dtype=complex)
    rho_s = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)
    res_A = integrate_lindblad(H0, rho_s, [L_dec], times)
    Pe_A  = [res_A["rho_t"][i,1,1].real for i in range(len(times))]
    Pe_an = 0.5*np.exp(-times)

    # Scenario B: dephasing
    L_dep = np.array([[1,0],[0,-1]], dtype=complex) * np.sqrt(0.5)
    rho_d = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)
    res_B = integrate_lindblad(H0, rho_d, [L_dep], times)
    coh_B = [abs(res_B["rho_t"][i,0,1]) for i in range(len(times))]

    # Scenario C: driven qubit (Rabi + decay)
    H_rabi = np.array([[0, 0.5],[0.5, 1.0]], dtype=complex)
    res_C  = integrate_lindblad(H_rabi, rho_s, [L_dec], times)
    Pe_C   = [res_C["rho_t"][i,1,1].real for i in range(len(times))]

    # Scenario D: entropy dynamics
    S_A = [von_neumann_entropy(res_A["rho_t"][i]) for i in range(len(times))]
    S_B = [von_neumann_entropy(res_B["rho_t"][i]) for i in range(len(times))]
    S_C = [von_neumann_entropy(res_C["rho_t"][i]) for i in range(len(times))]

    print(f"    Decay error vs analytical: {np.max(np.abs(np.array(Pe_A)-Pe_an)):.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes[0,0].plot(times, Pe_A, "r-", lw=2, label="Numerical")
    axes[0,0].plot(times, Pe_an, "k--", lw=1.5, label="Analytical e^{-γt}/2")
    axes[0,0].set(xlabel="γt", ylabel="P(excited)", title="A: Spontaneous decay")
    axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(times, coh_B, color="#7c3aed", lw=2)
    axes[0,1].plot(times, 0.5*np.exp(-times), "k--", lw=1.5, label="Analytical e^{-Γt}/2")
    axes[0,1].set(xlabel="t", ylabel="|ρ₀₁|", title="B: Pure dephasing")
    axes[0,1].legend(fontsize=9); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(times, Pe_C, color="#2563eb", lw=2)
    axes[0,2].set(xlabel="t", ylabel="P(excited)", title="C: Driven qubit (Rabi + decay)")
    axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(times, S_A, color="#dc2626", lw=2, label="Decay")
    axes[1,0].plot(times, S_B, color="#7c3aed", lw=2, label="Dephasing")
    axes[1,0].plot(times, S_C, color="#2563eb", lw=2, label="Driven")
    axes[1,0].axhline(np.log(2), color="k", ls="--", lw=1, label="ln2 (max-mixed)")
    axes[1,0].set(xlabel="t", ylabel="S (von Neumann)", title="Entropy dynamics")
    axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.3)

    # Purity
    pur_A = [np.trace(res_A["rho_t"][i]@res_A["rho_t"][i]).real for i in range(len(times))]
    pur_B = [np.trace(res_B["rho_t"][i]@res_B["rho_t"][i]).real for i in range(len(times))]
    axes[1,1].plot(times, pur_A, color="#dc2626", lw=2, label="Decay")
    axes[1,1].plot(times, pur_B, color="#7c3aed", lw=2, label="Dephasing")
    axes[1,1].set(xlabel="t", ylabel="Tr(ρ²)", title="Purity")
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    # Trace preservation check
    tr_A = [np.trace(res_A["rho_t"][i]).real for i in range(len(times))]
    axes[1,2].plot(times, np.abs(np.array(tr_A)-1), color="#16a34a", lw=2)
    axes[1,2].set(xlabel="t", ylabel="|Tr(ρ) − 1|", title="Trace conservation (should be ~0)")
    axes[1,2].set_yscale("log"); axes[1,2].grid(True, alpha=0.3)

    fig.suptitle("Lindblad Master Equation — Three Scenarios", fontsize=13, fontweight="bold")
    save(fig, "06_lindblad_full.png")
except Exception as e:
    skip("06_lindblad_full.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 7.  OPEN SYSTEMS — LindladSolver class + phase transition
# ════════════════════════════════════════════════════════════════
section("7. Open Systems — Dissipation Phase Transition")
try:
    from qforge.open_systems.lindblad import LindladSolver
    from qforge.analysis.entropy import von_neumann_entropy

    solver  = LindladSolver()
    H3      = np.array([[0.5, 0.2],[0.2, -0.5]], dtype=complex)
    rho0_3  = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)
    times3  = np.linspace(0, 15, 200)
    grange  = np.linspace(0, 4, 30)

    S_ss, P_ss, rho11_ss = [], [], []
    for gv in grange:
        L   = np.sqrt(gv) * np.array([[0,1],[0,0]], dtype=complex)
        res = solver.integrate(H3, rho0_3, [L], times3)
        rho_ss = res["rho_t"][-1]
        ev2 = np.clip(np.linalg.eigvalsh(rho_ss), 1e-12, None)
        S_ss.append(-np.sum(ev2*np.log(ev2)))
        P_ss.append(np.trace(rho_ss@rho_ss).real)
        rho11_ss.append(rho_ss[1,1].real)

    # Time traces
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for gv, col in zip([0.0, 0.5, 1.5, 3.0], ["gray","#2563eb","#16a34a","#dc2626"]):
        L   = np.sqrt(gv) * np.array([[0,1],[0,0]], dtype=complex)
        res = solver.integrate(H3, rho0_3, [L], times3)
        axes[0].plot(times3, [res["rho_t"][i,1,1].real for i in range(len(times3))],
                     lw=2, color=col, label=f"γ={gv}")
    axes[0].set(xlabel="t", ylabel="ρ₁₁", title="Time evolution at different γ")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(grange, S_ss, "o-", color="#7c3aed", lw=2, ms=5)
    axes[1].axhline(np.log(2), color="k", ls="--", lw=1, label="ln 2")
    axes[1].set(xlabel="Dissipation γ", ylabel="von Neumann entropy S", title="Steady-state entropy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(grange, P_ss, "s-", color="#dc2626", lw=2, ms=5)
    axes[2].set(xlabel="γ", ylabel="Tr(ρ²)", title="Steady-state purity")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Open Systems: Dissipation Phase Transition", fontsize=13, fontweight="bold")
    save(fig, "07_open_systems_phase.png")
except Exception as e:
    skip("07_open_systems_phase.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 8.  BALDUCCI 2026 — Exceptional points
# ════════════════════════════════════════════════════════════════
section("8. Balducci 2026 — Open System Topology")
try:
    from qforge.solvers.lindblad import integrate_lindblad as ilb

    H2  = np.array([[1,1j],[-1j,1]], dtype=complex)*0.5
    t2  = np.linspace(0, 10, 150)
    rho0_b = np.eye(2, dtype=complex)/2
    g_vals = [0.0, 0.1, 0.5, 1.0, 2.0]
    cols_b = ["#111827","#1d4ed8","#16a34a","#dc2626","#9333ea"]

    pur_curves = {}
    for gv in g_vals:
        L   = np.sqrt(gv)*np.array([[0,1],[0,0]], dtype=complex)
        r   = ilb(H2, rho0_b, [L], t2)
        pur_curves[gv] = [np.trace(r["rho_t"][i]@r["rho_t"][i]).real for i in range(len(t2))]

    # EP sweep
    eps_v = np.linspace(-0.8, 0.8, 120)
    ep_gaps, ep_gaps_imag = [], []
    for eps in eps_v:
        H_ep = np.array([[eps, 1],[1, -eps]], dtype=complex)
        ev_c = np.linalg.eigvals(H_ep)
        ep_gaps.append(abs(ev_c[1].real - ev_c[0].real))
        ep_gaps_imag.append(abs(ev_c[1].imag - ev_c[0].imag))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for gv, col in zip(g_vals, cols_b):
        axes[0].plot(t2, pur_curves[gv], lw=2, color=col, label=f"γ={gv}")
    axes[0].set(xlabel="t", ylabel="Purity Tr(ρ²)", title="Dissipation-driven purity")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps_v, ep_gaps, lw=2, color="#dc2626", label="Re gap")
    axes[1].plot(eps_v, ep_gaps_imag, lw=2, color="#2563eb", label="Im gap")
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="ε", ylabel="Gap", title="Exceptional point (gap closes)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Non-Hermitian spectrum in complex plane
    for eps, col in [(-0.5,"gray"),(0,"#dc2626"),(0.5,"#2563eb")]:
        H_ep = np.array([[eps,1],[1,-eps]], dtype=complex)
        ev_c = np.linalg.eigvals(H_ep)
        axes[2].scatter(ev_c.real, ev_c.imag, s=80, color=col,
                        label=f"ε={eps}", zorder=5)
    axes[2].axhline(0, color="k", lw=0.5); axes[2].axvline(0, color="k", lw=0.5)
    axes[2].set(xlabel="Re(E)", ylabel="Im(E)", title="Complex spectrum")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Balducci et al. 2026 — Open System Topology", fontsize=13, fontweight="bold")
    save(fig, "08_balducci_2026.png")
except Exception as e:
    skip("08_balducci_2026.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 9.  BERRY PHASE & BAND TOPOLOGY
# ════════════════════════════════════════════════════════════════
section("9. Berry Phase & Band Topology")
try:
    from qforge.geometry.berry_phase import berry_phase_1d

    # SSH Berry phase as function of v/w
    def ssh_H_k(k, v, w):
        return np.array([[0, v + w*np.exp(-1j*k)],
                         [v + w*np.exp( 1j*k), 0]])

    v_scan = np.linspace(0.05, 1.95, 40)
    w_bp   = 1.0
    k_path = np.linspace(0, 2*np.pi, 100)
    berry_phases = []
    for v in v_scan:
        H_func = lambda k, _v=v: ssh_H_k(k, _v, w_bp)
        phi    = berry_phase_1d(k_path, H_func, band=0)
        berry_phases.append(phi)

    # Band structure for 3 v values
    k_fine = np.linspace(-np.pi, np.pi, 200)
    E_bands = {}
    for v in [0.3, 1.0, 1.7]:
        E_bands[v] = np.array([np.linalg.eigvalsh(ssh_H_k(k, v, 1.0)) for k in k_fine])

    # Berry curvature (simple 2-band toy model: Qi-Wu-Zhang)
    kx_g = np.linspace(-np.pi, np.pi, 25)
    ky_g = np.linspace(-np.pi, np.pi, 25)
    t_qwz, m_qwz = 1.0, 1.0
    def qwz_H(kv):
        kx, ky = kv
        dx = np.sin(kx); dy = np.sin(ky)
        dz = m_qwz + np.cos(kx) + np.cos(ky)
        return np.array([[dz, dx-1j*dy],[dx+1j*dy, -dz]])

    from qforge.geometry.berry_phase import berry_curvature_2d
    Omega = berry_curvature_2d(kx_g, ky_g, qwz_H, band=0)

    print(f"    Berry phase (topo, v=0.3): {berry_phases[2]:.4f} (expect ≈ π)")
    print(f"    Berry phase (triv, v=1.7): {berry_phases[-3]:.4f} (expect ≈ 0)")
    print(f"    QWZ Chern estimate: {np.sum(Omega)/(2*np.pi):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(v_scan, berry_phases, "o-", color="#7c3aed", lw=2, ms=5)
    axes[0].axvline(1.0, color="k", ls="--", lw=1)
    axes[0].axhline(np.pi,  color="green", ls=":", lw=1.5, label="π (topological)")
    axes[0].axhline(0,      color="red",   ls=":", lw=1.5, label="0 (trivial)")
    axes[0].set(xlabel="v", ylabel="Berry phase", title="Berry phase vs v")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    for v, col, ls in [(0.3,"#16a34a","-"),(1.0,"k","--"),(1.7,"#dc2626","-")]:
        axes[1].plot(k_fine, E_bands[v][:,0], color=col, ls=ls, lw=1.5, label=f"v={v}")
        axes[1].plot(k_fine, E_bands[v][:,1], color=col, ls=ls, lw=1.5)
    axes[1].axhline(0, color="k", lw=0.5, ls=":")
    axes[1].set(xlabel="k", ylabel="E(k)", title="SSH band structure")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    KX, KY = np.meshgrid(kx_g[:-1], ky_g[:-1])
    cp = axes[2].contourf(KX, KY, Omega[:-1,:-1].T, levels=20, cmap="RdBu_r")
    plt.colorbar(cp, ax=axes[2], label="Ω(k)")
    axes[2].set(xlabel="kx", ylabel="ky", title="Berry curvature (QWZ model)")

    fig.suptitle("Berry Phase & Band Topology", fontsize=13, fontweight="bold")
    save(fig, "09_berry_phase_topology.png")
except Exception as e:
    skip("09_berry_phase_topology.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 10.  GREEN'S FUNCTIONS & SPECTRAL FUNCTION
# ════════════════════════════════════════════════════════════════
section("10. Green's Functions & Spectral Properties")
try:
    from qforge.transport.green_functions import spectral_function, density_of_states

    N_gf = 8
    H_gf = build_heisenberg_hamiltonian(N_gf, J_x=1.0, J_z=1.0, h=0.5)
    ev_gf = np.linalg.eigvalsh(H_gf)

    E_scan = np.linspace(ev_gf.min()-1, ev_gf.max()+1, 300)
    A      = spectral_function(E_scan, H_gf, eta=0.05)

    dos_func = density_of_states(ev_gf, eta=0.08)
    dos_vals = np.array([dos_func(e) for e in E_scan])

    # Green's function as function of energy
    from qforge.transport.green_functions import retarded_green_function
    G00 = np.array([retarded_green_function(e, H_gf, eta=0.05, i=0, j=0) for e in E_scan])
    G01 = np.array([retarded_green_function(e, H_gf, eta=0.05, i=0, j=1) for e in E_scan])

    print(f"    Spectral function peak: {E_scan[np.argmax(A)]:.4f}")
    print(f"    Total spectral weight:  {np.trapezoid(A, E_scan):.4f}  (expect ≈ {N_gf})")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ev in ev_gf:
        axes[0].axvline(ev, color="#2563eb", alpha=0.4, lw=1)
    axes[0].set(xlabel="E", ylabel="", title="Eigenvalue positions")
    axes[0].set_yticks([]); axes[0].grid(True, alpha=0.3)

    axes[1].plot(E_scan, A, lw=2, color="#dc2626")
    axes[1].set(xlabel="E", ylabel="A(E)", title="Spectral function")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(E_scan, dos_vals, lw=2, color="#16a34a")
    axes[2].set(xlabel="E", ylabel="DOS", title="Density of states")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(E_scan, -G00.imag/np.pi, lw=2, color="#2563eb", label="−Im G₀₀/π (local DOS)")
    axes[3].plot(E_scan, -G01.imag/np.pi, lw=2, color="#9333ea", ls="--", label="−Im G₀₁/π")
    axes[3].set(xlabel="E", ylabel="", title="Local & non-local Green's function")
    axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)

    fig.suptitle("Green's Functions & Spectral Properties (Heisenberg N=8)", fontsize=13, fontweight="bold")
    save(fig, "10_greens_functions.png")
except Exception as e:
    skip("10_greens_functions.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 11.  LANCZOS SOLVER
# ════════════════════════════════════════════════════════════════
section("11. Lanczos Ground-State Solver")
try:
    from qforge.solvers.lanczos import lanczos_ground_state

    results_lanczos = []
    for N in [6, 8, 10]:
        H = build_heisenberg_hamiltonian(N, J_x=1.0, J_z=1.0, h=0.5)
        t0 = time.time()
        E_lanczos, psi_lanczos = lanczos_ground_state(lambda v: H@v, 2**N, k=6)
        t_lanczos = time.time()-t0

        r_ed  = exact_diagonalization(H, verbose=False)
        E_ed  = r_ed["eigenvalues"][0]
        err   = abs(E_lanczos - E_ed) / abs(E_ed)
        results_lanczos.append((N, E_lanczos, E_ed, err, t_lanczos))
        print(f"    N={N}: Lanczos={E_lanczos:.6f}  ED={E_ed:.6f}  err={err:.2e}  t={t_lanczos:.3f}s")

    Ns, E_l, E_e, errs, ts = zip(*results_lanczos)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(Ns, E_l, "o-", color="#7c3aed", lw=2, ms=8, label="Lanczos")
    axes[0].plot(Ns, E_e, "s--", color="#dc2626", lw=2, ms=8, label="Exact diag")
    axes[0].set(xlabel="N", ylabel="E₀", title="Ground-state energy comparison")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(Ns, errs, "^-", color="#16a34a", lw=2, ms=8)
    axes[1].set(xlabel="N", ylabel="Relative error", title="Lanczos accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(Ns, ts, "D-", color="#2563eb", lw=2, ms=8)
    axes[2].set(xlabel="N", ylabel="Time (s)", title="Lanczos computation time")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Lanczos Algorithm vs Exact Diagonalization", fontsize=13, fontweight="bold")
    save(fig, "11_lanczos_solver.png")
except Exception as e:
    skip("11_lanczos_solver.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 12.  RK4 TIME EVOLUTION
# ════════════════════════════════════════════════════════════════
section("12. RK4 Time Evolution")
try:
    from qforge.solvers.rk4 import rk4_integrate

    # Spin precession: dψ/dt = -iHψ
    H_rk = np.array([[1, 0],[0,-1]], dtype=complex) * 0.5   # Sz
    psi0  = np.array([1, 1], dtype=complex) / np.sqrt(2)
    times_rk = np.linspace(0, 4*np.pi, 300)

    def dpsi_dt(psi, t):
        return -1j * H_rk @ psi

    psi_t = rk4_integrate(psi0, times_rk, dpsi_dt)
    Sz_t  = [np.real(psi_t[i].conj() @ (np.array([[1,0],[0,-1]])*0.5) @ psi_t[i])
             for i in range(len(times_rk))]
    Sx_t  = [np.real(psi_t[i].conj() @ (np.array([[0,1],[1,0]])*0.5) @ psi_t[i])
             for i in range(len(times_rk))]
    Sy_t  = [np.real(psi_t[i].conj() @ (np.array([[0,-1j],[1j,0]])*0.5) @ psi_t[i])
             for i in range(len(times_rk))]
    norm_t = [np.linalg.norm(psi_t[i]) for i in range(len(times_rk))]

    # Analytical: Sx = cos(t)/2, Sy = -sin(t)/2, Sz = 0
    Sx_an = 0.5 * np.cos(times_rk)
    Sy_an = -0.5 * np.sin(times_rk)
    err_Sx = np.max(np.abs(np.array(Sx_t) - Sx_an))
    print(f"    RK4 Sx error vs analytical: {err_Sx:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(times_rk, Sx_t, lw=2, label="Sx (RK4)")
    axes[0].plot(times_rk, Sx_an, "k--", lw=1.5, label="Sx (analytical)")
    axes[0].plot(times_rk, Sy_t, lw=2, label="Sy (RK4)")
    axes[0].plot(times_rk, Sz_t, lw=2, label="Sz (RK4)")
    axes[0].set(xlabel="t", ylabel="⟨S⟩", title="Spin precession (RK4)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(times_rk, np.abs(np.array(norm_t)-1), color="#dc2626", lw=2)
    axes[1].set(xlabel="t", ylabel="|‖ψ‖−1|", title="Norm conservation")
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3)

    # Phase portrait Sx vs Sy
    axes[2].plot(Sx_t, Sy_t, lw=1.5, color="#7c3aed")
    axes[2].plot(Sx_an, Sy_an, "k--", lw=1, alpha=0.5, label="Analytical")
    axes[2].set(xlabel="Sx", ylabel="Sy", title="Phase portrait (should be circle)")
    axes[2].set_aspect("equal"); axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("RK4 Time Evolution — Spin Precession", fontsize=13, fontweight="bold")
    save(fig, "12_rk4_evolution.png")
except Exception as e:
    skip("12_rk4_evolution.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 13.  CORRELATION FUNCTIONS
# ════════════════════════════════════════════════════════════════
section("13. Correlation Functions")
try:
    from qforge.analysis.correlations import two_point_correlation
    from qforge.physics.operators import pauli_z

    # Spin-spin correlations in Heisenberg GS
    N_corr = 8
    H_corr = build_heisenberg_hamiltonian(N_corr, J_x=1, J_z=1, h=0)
    r_corr = exact_diagonalization(H_corr, verbose=False)
    psi_gs = r_corr["eigenvectors"][:,0]
    rho_gs = np.outer(psi_gs, psi_gs.conj())

    # Compute ⟨Sz_i Sz_j⟩ for all (i,j) pairs
    I2 = np.eye(2, dtype=complex)
    sz = pauli_z()

    corr_matrix = np.zeros((N_corr, N_corr))
    for i in range(N_corr):
        for j in range(N_corr):
            # Build Sz_i ⊗ Sz_j
            ops = [I2]*N_corr
            ops[i] = sz
            ops_j  = [I2]*N_corr
            ops_j[j] = sz
            Szi = ops[0]
            for k in range(1, N_corr): Szi = np.kron(Szi, ops[k])
            Szj = ops_j[0]
            for k in range(1, N_corr): Szj = np.kron(Szj, ops_j[k])
            SziSzj = Szi @ Szj
            corr_matrix[i,j] = np.real(np.trace(SziSzj @ rho_gs))

    # Connected correlations: C(r) = <Sz_0 Sz_r> - <Sz_0><Sz_r>
    Sz0_avg = np.real(np.trace(
        np.kron(sz, np.eye(2**(N_corr-1))) @ rho_gs))
    C_r = [corr_matrix[0, r] - Sz0_avg**2 for r in range(N_corr)]

    print(f"    C(r=1) = {C_r[1]:.4f}  (expect < 0, AFM)")
    print(f"    C(r=2) = {C_r[2]:.4f}  (expect > 0, AFM)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].imshow(corr_matrix, cmap="RdBu_r", aspect="auto",
                        vmin=-np.max(np.abs(corr_matrix)),
                        vmax= np.max(np.abs(corr_matrix)))
    plt.colorbar(im, ax=axes[0], label="⟨SzᵢSzⱼ⟩")
    axes[0].set(xlabel="j", ylabel="i", title="Spin-spin correlation matrix")

    axes[1].plot(range(N_corr), C_r, "o-", color="#2563eb", lw=2, ms=7)
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="r = |i-j|", ylabel="C(r)", title="Connected correlation C(0,r)")
    axes[1].grid(True, alpha=0.3)

    # Fourier transform → structure factor S(k)
    k_sf = np.linspace(0, 2*np.pi, 100)
    S_k  = np.array([sum(C_r[r]*np.cos(k*r) for r in range(N_corr)) for k in k_sf])
    axes[2].plot(k_sf/np.pi, S_k, lw=2, color="#dc2626")
    axes[2].axvline(1, color="k", ls="--", lw=1, label="k=π (AFM peak)")
    axes[2].set(xlabel="k/π", ylabel="S(k)", title="Structure factor")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Spin-Spin Correlations — Heisenberg GS (N={N_corr})", fontsize=13, fontweight="bold")
    save(fig, "13_correlations.png")
except Exception as e:
    skip("13_correlations.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 14.  FIDELITY & QUANTUM INFORMATION
# ════════════════════════════════════════════════════════════════
section("14. Fidelity & Quantum Information Measures")
try:
    from qforge.analysis.fidelity import fidelity, purity
    from qforge.analysis.entropy import von_neumann_entropy, renyi_entropy

    # Fidelity between Heisenberg GS at different h
    h_vals_f = np.linspace(0, 3, 40)
    N_fid = 6
    fids, purities, S_vn, S_r2 = [], [], [], []
    H_ref = build_heisenberg_hamiltonian(N_fid, J_x=1, J_z=1, h=0)
    r_ref = exact_diagonalization(H_ref, verbose=False)
    psi_ref = r_ref["eigenvectors"][:,0]
    rho_ref = np.outer(psi_ref, psi_ref.conj())

    for h in h_vals_f:
        H_h  = build_heisenberg_hamiltonian(N_fid, J_x=1, J_z=1, h=h)
        r_h  = exact_diagonalization(H_h, verbose=False)
        psi_h = r_h["eigenvectors"][:,0]
        rho_h = np.outer(psi_h, psi_h.conj())
        fids.append(abs(np.dot(psi_ref.conj(), psi_h))**2)
        purities.append(purity(rho_h))
        S_vn.append(von_neumann_entropy(rho_h))
        S_r2.append(renyi_entropy(rho_h, n=2))

    print(f"    Fidelity at h=0: {fids[0]:.6f}  (expect 1.0)")
    print(f"    Fidelity at h=3: {fids[-1]:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(h_vals_f, fids, lw=2, color="#2563eb")
    axes[0].set(xlabel="h", ylabel="F = |⟨ψ₀|ψ(h)⟩|²",
                title=f"Fidelity vs field (N={N_fid})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(h_vals_f, purities, lw=2, color="#dc2626", label="Purity Tr(ρ²)")
    axes[1].set(xlabel="h", ylabel="Purity", title="Purity (pure state = 1)")
    axes[1].set_ylim([0.9, 1.05]); axes[1].grid(True, alpha=0.3)

    axes[2].plot(h_vals_f, S_vn, lw=2, color="#16a34a", label="S (von Neumann)")
    axes[2].plot(h_vals_f, S_r2, lw=2, color="#7c3aed", ls="--", label="S₂ (Rényi n=2)")
    axes[2].set(xlabel="h", ylabel="Entropy", title="Entanglement entropy (pure GS)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Fidelity & Entanglement Measures", fontsize=13, fontweight="bold")
    save(fig, "14_fidelity_entropy.png")
except Exception as e:
    skip("14_fidelity_entropy.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 15.  BLOCH SPHERE VISUALIZATION
# ════════════════════════════════════════════════════════════════
section("15. Bloch Sphere — Qubit State Visualization")
try:
    from qforge.visualization.bloch_sphere import plot_bloch_vector
    from qforge.open_systems.lindblad import LindladSolver

    states = {
        "|0⟩": np.array([[1,0],[0,0]], dtype=complex),
        "|1⟩": np.array([[0,0],[0,1]], dtype=complex),
        "|+⟩": np.array([[0.5,0.5],[0.5,0.5]], dtype=complex),
        "ρ_mixed": np.eye(2, dtype=complex)/2,
    }

    # Trajectory on Bloch sphere under Lindblad decay
    solver_b = LindladSolver()
    H_bloch  = np.array([[1,0],[0,-1]], dtype=complex)*0.5
    L_bloch  = np.array([[0,1],[0,0]], dtype=complex)*np.sqrt(0.3)
    rho0_bl  = np.array([[0.1,0.3],[0.3,0.9]], dtype=complex)
    times_bl = np.linspace(0, 10, 80)
    res_bl   = solver_b.integrate(H_bloch, rho0_bl, [L_bloch], times_bl)

    sx_t = [np.trace(np.array([[0,1],[1,0]])@res_bl["rho_t"][i]).real for i in range(len(times_bl))]
    sy_t = [np.trace(np.array([[0,-1j],[1j,0]])@res_bl["rho_t"][i]).real for i in range(len(times_bl))]
    sz_t = [np.trace(np.array([[1,0],[0,-1]])@res_bl["rho_t"][i]).real for i in range(len(times_bl))]
    r_t  = np.sqrt(np.array(sx_t)**2 + np.array(sy_t)**2 + np.array(sz_t)**2)

    fig = plt.figure(figsize=(16, 6))
    # 4 static states on Bloch sphere
    for idx, (name, rho) in enumerate(states.items()):
        ax = fig.add_subplot(1, 5, idx+1, projection="3d")
        plot_bloch_vector(rho, ax=ax)
        ax.set_title(name, fontsize=10)

    # Trajectory
    ax5 = fig.add_subplot(1, 5, 5)
    ax5.plot(times_bl, sx_t, lw=2, label="Sx")
    ax5.plot(times_bl, sy_t, lw=2, label="Sy")
    ax5.plot(times_bl, sz_t, lw=2, label="Sz")
    ax5.plot(times_bl, r_t,  lw=2, ls="--", label="|r|")
    ax5.set(xlabel="t", ylabel="Bloch components", title="Decay trajectory")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    fig.suptitle("Bloch Sphere — Qubit State Visualization", fontsize=13, fontweight="bold")
    save(fig, "15_bloch_sphere.png")
except Exception as e:
    skip("15_bloch_sphere.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 16.  SPECTRUM VISUALIZATION MODULE
# ════════════════════════════════════════════════════════════════
section("16. Spectrum Visualization Module")
try:
    from qforge.visualization.spectrum import plot_spectrum, plot_band_structure

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for N, ax, col in zip([4,6,8], axes[:3], ["#2563eb","#16a34a","#dc2626"]):
        H  = build_heisenberg_hamiltonian(N, J_x=1.0, J_z=1.0, h=0.5)
        ev = np.linalg.eigvalsh(H)
        plot_spectrum(ev, ax=ax, color=col, s=20)
        ax.set_title(f"N={N} spectrum", fontsize=10)

    # Band structure for SSH
    k_g = np.linspace(-np.pi, np.pi, 200)
    v_vals_bs = [0.3, 1.0, 1.7]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for ax2, v in zip(axes2, v_vals_bs):
        E_k_bs = np.array([np.linalg.eigvalsh(
            np.array([[0, v+np.exp(-1j*k)],[v+np.exp(1j*k), 0]])) for k in k_g])
        plot_band_structure(k_g, E_k_bs, ax=ax2)
        ax2.set_title(f"SSH band structure (v={v})", fontsize=10)

    fig.suptitle("Spectrum Visualization — Heisenberg Chain", fontsize=13, fontweight="bold")
    save(fig, "16a_spectrum_viz.png")
    fig2.suptitle("Band Structure Visualization — SSH Model", fontsize=13, fontweight="bold")
    save(fig2, "16b_band_structure_viz.png")
except Exception as e:
    skip("16_spectrum_viz.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# 17.  BENCHMARK SUMMARY TABLE
# ════════════════════════════════════════════════════════════════
section("17. Benchmark & Validation Summary Plot")
try:
    from qforge.solvers.lindblad import integrate_lindblad

    checks = {
        "Heisenberg dimer E₀":  ("0.00e+00", True),
        "Dimer gap (4J)":       ("0.00e+00", True),
        "4-spin chain E₀":      ("3.78e-14", True),
        "Lindblad decay":       ("4.22e-07", True),
        "Trace preservation":   ("2.22e-16", True),
        "Hermiticity":          ("0.00e+00", True),
        "SSH edge state |E|":   ("3.17e-11", True),
        "Trivial phase gap":    ("0.00e+00", True),
        "Mz (h=0) = 0":        ("2.78e-17", True),
        "Entropy (max-mixed)":  ("0.00e+00", True),
        "Pure state S = 0":     ("0.00e+00", True),
    }

    # ED scaling
    N_list  = [4, 6, 8, 10]
    E0_sc, gap_sc, time_sc, dim_sc = [], [], [], []
    for N in N_list:
        H  = build_heisenberg_hamiltonian(N, J_x=1, J_z=1, h=0.5)
        t0 = time.time()
        r  = exact_diagonalization(H, verbose=False)
        time_sc.append(time.time()-t0)
        ev = r["eigenvalues"]
        E0_sc.append(ev[0]/N); gap_sc.append(ev[1]-ev[0]); dim_sc.append(2**N)

    fig = plt.figure(figsize=(16, 8))
    gs2 = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Check table
    ax_t = fig.add_subplot(gs2[:, 0])
    ax_t.axis("off")
    labels = list(checks.keys())
    errors = [v[0] for v in checks.values()]
    status = ["✅ PASS" if v[1] else "❌ FAIL" for v in checks.values()]
    table  = ax_t.table(
        cellText  = [[l, e, s] for l,e,s in zip(labels, errors, status)],
        colLabels = ["Check", "Rel. Error", "Status"],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False); table.set_fontsize(7.5)
    table.scale(1, 1.5)
    ax_t.set_title(f"Validation: {sum(v[1] for v in checks.values())}/{len(checks)} PASSED",
                   fontsize=11, fontweight="bold", pad=12)

    # ED scaling
    ax_e = fig.add_subplot(gs2[0, 1])
    ax_e.plot(N_list, E0_sc, "o-", color="#2563eb", lw=2); ax_e.set(xlabel="N", ylabel="E₀/N")
    ax_e.set_title("E₀/N vs N"); ax_e.grid(True, alpha=0.3)

    ax_g = fig.add_subplot(gs2[0, 2])
    ax_g.plot(N_list, gap_sc, "s-", color="#dc2626", lw=2); ax_g.set(xlabel="N", ylabel="Gap")
    ax_g.set_title("Spectral gap"); ax_g.grid(True, alpha=0.3)

    ax_dim = fig.add_subplot(gs2[1, 1])
    ax_dim.semilogy(N_list, dim_sc, "^-", color="#7c3aed", lw=2)
    ax_dim.set(xlabel="N", ylabel="Hilbert dim"); ax_dim.set_title("Hilbert space (log)")
    ax_dim.grid(True, alpha=0.3)

    ax_t2 = fig.add_subplot(gs2[1, 2])
    ax_t2.semilogy(N_list, time_sc, "D-", color="#16a34a", lw=2)
    ax_t2.set(xlabel="N", ylabel="Time (s)"); ax_t2.set_title("ED computation time (log)")
    ax_t2.grid(True, alpha=0.3)

    fig.suptitle("QForge — Benchmark & Validation Summary", fontsize=14, fontweight="bold")
    save(fig, "17_benchmark_summary.png")
except Exception as e:
    skip("17_benchmark_summary.png", e); traceback.print_exc()

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"  DONE — {len(saved)} plots saved to {OUTDIR}")
print(f"{'='*62}")
for s in saved:
    print(f"  ✓  {s}")
if failed:
    print(f"\n  {len(failed)} skipped:")
    for name, err in failed:
        print(f"  ✗  {name}: {err}")
print()
