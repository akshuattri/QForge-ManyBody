"""Non-equilibrium Green's functions."""
import numpy as np

def lesser_green_function(H: np.ndarray, Sigma_lesser: np.ndarray,
                         E: float, eta: float = 1e-3) -> np.ndarray:
    """Lesser Green's function for NEGF."""
    
    H_E = (E + 1j*eta)*np.eye(len(H)) - H
    G_r = np.linalg.inv(H_E)
    
    G_lesser = G_r @ Sigma_lesser @ G_r.conj().T
    
    return G_lesser

def transport_current(H_left: np.ndarray, H_right: np.ndarray,
                     H_center: np.ndarray, V: np.ndarray,
                     mu_L: float, mu_R: float, T: float = 0.0) -> float:
    """Compute transport current (simplified)."""
    
    eV = mu_L - mu_R
    
    # Simplified conductance
    G_0 = 2 * np.pi  # Conductance quantum
    
    # Transmission probability (simplified)
    T_trans = np.abs(V) ** 2
    
    if T > 0:
        # Thermal broadening
        factor = 1.0 / (1 + np.exp(eV / T))
    else:
        factor = 1.0 if eV > 0 else 0.0
    
    return G_0 * T_trans * factor * eV
