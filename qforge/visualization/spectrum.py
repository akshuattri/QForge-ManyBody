"""Spectrum visualization."""
import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(eigenvalues: np.ndarray, ax=None, **kwargs):
    """Plot eigenvalue spectrum."""
    if ax is None:
        ax = plt.gca()
    
    sorted_eigs = np.sort(eigenvalues)
    indices = np.arange(len(sorted_eigs))
    
    ax.scatter(indices, sorted_eigs, **kwargs)
    ax.set_xlabel("State Index", fontsize=12)
    ax.set_ylabel("Energy", fontsize=12)
    ax.grid(True, alpha=0.3)
    return ax

def plot_band_structure(k_grid: np.ndarray, E_k: np.ndarray, ax=None):
    """Plot band structure."""
    if ax is None:
        ax = plt.gca()
    
    if E_k.ndim == 1:
        ax.plot(k_grid, E_k, 'b-', linewidth=2)
    else:
        for n in range(E_k.shape[1]):
            ax.plot(k_grid, E_k[:, n], 'b-', linewidth=2)
    
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("E(k)", fontsize=12)
    ax.grid(True, alpha=0.3)
    return ax
