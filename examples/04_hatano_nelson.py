"""Example 4: Hatano-Nelson — Non-Hermitian Skin Effect"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qforge.models.hatano_nelson import build_hatano_nelson

N = 40

# ── Spectrum vs asymmetry ──────────────────────────────────────────────────
alphas = np.linspace(0.0, 0.5, 30)
spectra = [np.linalg.eigvalsh(build_hatano_nelson(N, alpha=a)) for a in alphas]

# ── Participation ratio (skin effect probe) ────────────────────────────────
alpha_demo = 0.3
_, evecs = np.linalg.eigh(build_hatano_nelson(N, alpha=alpha_demo))
PR = [1.0 / np.sum(np.abs(evecs[:, i])**4) for i in range(N)]

print("Hatano-Nelson Non-Hermitian Model")
print(f"  N = {N} sites,  α = {alpha_demo}")
print(f"  Avg participation ratio : {np.mean(PR):.2f}  (bulk ~ N/2 = {N//2})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for i, (a, spec) in enumerate(zip(alphas, spectra)):
    ax1.plot([a]*N, spec, ".", ms=2, color=plt.cm.viridis(i / len(alphas)), alpha=0.6)
ax1.set(xlabel="Asymmetry α", ylabel="Energy", title="Spectrum vs asymmetry")
ax1.grid(True, alpha=0.3)

ax2.bar(range(N), PR, color="#7c3aed", alpha=0.8)
ax2.axhline(N / 4, color="k", ls="--", lw=1, label=f"N/4 = {N//4}")
ax2.set(xlabel="Eigenstate index", ylabel="Participation ratio",
        title=f"Skin effect probe (α={alpha_demo})")
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle("Hatano-Nelson Non-Hermitian Chain", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("hatano_nelson.png", dpi=150, bbox_inches="tight")
print("  Saved → hatano_nelson.png")
