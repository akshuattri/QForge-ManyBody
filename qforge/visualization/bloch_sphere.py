"""Bloch sphere visualization."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_bloch_vector(rho: np.ndarray, ax=None):
    """Plot state on Bloch sphere."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Bloch vector components
    sx = np.trace(np.array([[0, 1], [1, 0]]) @ rho).real
    sy = np.trace(np.array([[0, -1j], [1j, 0]]) @ rho).real
    sz = np.trace(np.array([[1, 0], [0, -1]]) @ rho).real
    
    # Draw Bloch sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='blue')
    
    # Draw Bloch vector
    ax.quiver(0, 0, 0, sx, sy, sz, color='red', arrow_length_ratio=0.1, linewidth=2)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax
