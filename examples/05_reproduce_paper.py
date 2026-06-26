"""Example 5: Reproduce Paper Results — SSH & Lindblad Phase Transition

Demonstrates QForge's paper-reproduction workflow by reproducing key results
from two reference calculations included in the papers/ directory.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=" * 60)
print("Paper Reproduction Demo")
print("=" * 60)

# ── 1. SSH topological model ───────────────────────────────────────────────
print("\n[1] SSH Topological Model — edge state verification")
from papers.ssh_topological.reproduce import main as ssh_main
ssh_main()

# ── 2. Lindblad dissipation phase transition ──────────────────────────────
print("\n[2] Lindblad Dynamics — dissipation-induced phase transition")
from papers.lindblad_dynamics.reproduce import main as lindblad_main
lindblad_main()

print("\n" + "=" * 60)
print("Reproduction complete.")
print("=" * 60)
