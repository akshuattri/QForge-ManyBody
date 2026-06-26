
"""Reproduce: SSH Topological Model - Edge States"""

import numpy as np

def ssh_edge_states():
    """Calculate SSH edge states."""
    
    print("SSH Topological Model: Edge State Analysis")
    
    from qforge.models.ssh import build_ssh_hamiltonian
    
    # Topological phase
    H_topo = build_ssh_hamiltonian(N=10, v=0.5, w=1.0)
    eigs_topo = np.linalg.eigvalsh(H_topo)
    
    # Trivial phase
    H_trivial = build_ssh_hamiltonian(N=10, v=1.0, w=0.5)
    eigs_trivial = np.linalg.eigvalsh(H_trivial)
    
    print(f"  Topological phase gap: {eigs_topo[10] - eigs_topo[9]:.4f}")
    print(f"  Trivial phase gap: {eigs_trivial[10] - eigs_trivial[9]:.4f}")
    
    # Edge states appear at zero energy in topological phase
    mid_gap_topo = eigs_topo[10]
    mid_gap_trivial = eigs_trivial[10]
    
    print(f"  Mid-gap energy (topo): {mid_gap_topo:.6f}")
    print(f"  Mid-gap energy (triv): {mid_gap_trivial:.6f}")

def main():
    ssh_edge_states()

if __name__ == "__main__":
    main()
