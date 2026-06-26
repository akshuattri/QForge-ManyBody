#!/usr/bin/env python3
"""Reproduce published papers."""

import sys
import argparse

parser = argparse.ArgumentParser(description="Reproduce papers with QForge")
parser.add_argument("paper", choices=["balducci", "ssh", "lindblad", "nonhermitian"],
                   help="Paper to reproduce")
args = parser.parse_args()

print("="*70)
print(f"Paper Reproduction: {args.paper}")
print("="*70)

if args.paper == "balducci":
    from papers.balducci_2026.reproduce import fig1_dissipation_effect
    results = fig1_dissipation_effect()
    
elif args.paper == "ssh":
    from papers.ssh_topological.reproduce import ssh_edge_states
    ssh_edge_states()
    
elif args.paper == "lindblad":
    from papers.lindblad_dynamics.reproduce import (
        fig1_dissipation_phase_transition,
        fig2_steady_state_properties
    )
    gamma_vals, phase_diag = fig1_dissipation_phase_transition()
    times, purities, entropies = fig2_steady_state_properties()
    
elif args.paper == "nonhermitian":
    from papers.nonhermitian_topology.reproduce import (
        fig1_exceptional_point_motion,
        fig2_non_hermitian_skin_effect
    )
    alphas, gaps = fig1_exceptional_point_motion()
    participation = fig2_non_hermitian_skin_effect()

print("\nReproduction complete!")
