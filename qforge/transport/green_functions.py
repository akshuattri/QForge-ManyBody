"""Green's functions and spectral properties."""
import numpy as np

def retarded_green_function(E: float, H: np.ndarray, eta: float = 1e-3,
                            i: int = 0, j: int = 0) -> complex:
    """Retarded Green's function G^R_{ij}(E) = 1/(E - H + iη)."""
    G = np.linalg.inv((E + 1j*eta)*np.eye(len(H)) - H)
    return G[i, j]

def spectral_function(E: np.ndarray, H: np.ndarray, eta: float = 1e-3) -> np.ndarray:
    """Spectral function A(E) = -Im(G(E))/π."""
    A = np.zeros(len(E))
    for k, e in enumerate(E):
        G_diag = np.diag(np.linalg.inv((e + 1j*eta)*np.eye(len(H)) - H))
        A[k] = -np.sum(G_diag.imag) / np.pi
    return A

def density_of_states(eigenvalues: np.ndarray, eta: float = 0.01) -> callable:
    """DOS from eigenvalues."""
    def dos(E):
        return np.sum(eta / (np.pi * ((E - eigenvalues)**2 + eta**2)))
    return dos
