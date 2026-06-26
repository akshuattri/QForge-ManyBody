"""Data export utilities."""
import numpy as np

def export_eigenvalues(eigenvalues: np.ndarray, filename: str):
    """Export eigenvalues to CSV."""
    np.savetxt(filename, eigenvalues, fmt="%.15e")

def export_density_matrix(rho: np.ndarray, filename: str):
    """Export density matrix."""
    np.save(filename, rho)

def export_trajectory(times: np.ndarray, states: np.ndarray, filename: str):
    """Export time evolution trajectory."""
    data = np.column_stack([times, states])
    np.savetxt(filename, data, fmt="%.15e")
