"""Validation against analytical solutions."""
import numpy as np

class AnalyticalBenchmarks:
    """Analytical solutions for validation."""
    
    @staticmethod
    def heisenberg_2spin_xxx() -> np.ndarray:
        """2-spin XXX eigenvalues."""
        return np.array([-0.75, -0.75, 0.25, 0.25])
    
    @staticmethod
    def heisenberg_3spin_xxx() -> np.ndarray:
        """3-spin XXX eigenvalues (Bethe ansatz)."""
        return np.array([-1.75, -0.75, -0.75, 0.25, 0.25, 0.75, 1.25, 1.75])
    
    @staticmethod
    def ising_critical_field(N: int = 10) -> float:
        """Transverse-field Ising critical field."""
        return 1.0
    
    @staticmethod
    def harmonic_oscillator(n: int) -> float:
        """HO eigenvalues E_n = ω(n + 1/2)."""
        return 1.0 * (n + 0.5)
    
    @staticmethod
    def hydrogen_atom_levels(n: int) -> float:
        """Hydrogen eigenvalues E_n = -13.6 eV / n²."""
        return -13.6 / (n**2)

def validate_against_analytical(computed: np.ndarray, 
                               analytical: np.ndarray,
                               rtol: float = 1e-10) -> Dict[str, float]:
    """Compare computed vs analytical."""
    
    comp_sorted = np.sort(computed)
    anal_sorted = np.sort(analytical)
    
    errors = np.abs(comp_sorted - anal_sorted)
    rel_errors = errors / (1 + np.abs(anal_sorted))
    
    return {
        "max_abs_error": float(np.max(errors)),
        "mean_abs_error": float(np.mean(errors)),
        "max_rel_error": float(np.max(rel_errors)),
        "agreement": bool(np.allclose(comp_sorted, anal_sorted, rtol=rtol)),
    }
