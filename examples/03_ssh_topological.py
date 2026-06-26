"""Example 3: SSH Model — Topological Phase Transition"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qforge.models.ssh import build_ssh_hamiltonian

N = 20   # unit cells → 2N sites

# ── Phase diagram ─────────────────────────────────────────────────────────
v_vals  = np.linspace(0.1, 1.9, 80)
w       = 1.0
mid_gaps = []
for v in v_vals:
    eigs = np.linalg.eigvalsh(build_ssh_hamiltonian(N, v=v, w=w))
    mid  = len(eigs) // 2
    mid_gaps.append(eigs[mid] - eigs[mid - 1])

# ── Edge-state wavefunctions ───────────────────────────────────────────────
eigs_t, evecs_t = np.linalg.eigh(build_ssh_hamiltonian(N, v=0.3, w=1.0))
n_sites = len(eigs_t)

print("SSH Model — Topological Phase Transition")
print(f"  Topological gap  (v=0.3, w=1.0) : {mid_gaps[10]:.4f}")
print(f"  Trivial     gap  (v=1.7, w=1.0) : {mid_gaps[-10]:.4f}")
print(f"  Edge-state energy (should ≈ 0)  : {eigs_t[N]:.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(v_vals, mid_gaps, lw=2, color="#2563eb")
ax1.axvline(1.0, color="k", ls="--", lw=1)
ax1.fill_between(v_vals, mid_gaps, where=v_vals < w, alpha=0.12, color="green", label="Topological")
ax1.fill_between(v_vals, mid_gaps, where=v_vals >= w, alpha=0.12, color="red",   label="Trivial")
ax1.set(xlabel="v (intra-cell hopping)", ylabel="Mid-gap ΔE", title="Phase diagram")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.bar(range(n_sites), np.abs(evecs_t[:, N - 1])**2, color="#16a34a", alpha=0.8, label="Edge −")
ax2.bar(range(n_sites), np.abs(evecs_t[:, N    ])**2, color="#dc2626", alpha=0.5, label="Edge +")
ax2.set(xlabel="Site", ylabel="|ψ|²", title="Topological edge states (v=0.3)")
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle("SSH Model", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("ssh_topological.png", dpi=150, bbox_inches="tight")
print("  Saved → ssh_topological.png")
