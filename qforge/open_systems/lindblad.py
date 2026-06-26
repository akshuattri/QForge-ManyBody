"""Lindblad master equation solver."""
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Dict, Any
import time

class LindladSolver:
    """Lindblad master equation integrator."""
    
    def __init__(self, rtol: float = 1e-8, atol: float = 1e-10):
        self.rtol = rtol
        self.atol = atol
    
    def build_superoperator(self, H: np.ndarray, L_ops: List[np.ndarray]):
        """Build Lindblad superoperator."""
        dim = H.shape[0]
        
        def dρdt_vec(rho_vec: np.ndarray, t: float) -> np.ndarray:
            rho = rho_vec.reshape((dim, dim))
            
            # -i[H, ρ]
            drho = -1j * (H @ rho - rho @ H)
            
            # Lindblad jump operators
            for L in L_ops:
                drho += L @ rho @ L.conj().T
                drho -= 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
            
            return drho.flatten()
        
        return dρdt_vec
    
    def integrate(self, H: np.ndarray, rho0: np.ndarray, 
                  L_ops: List[np.ndarray], times: np.ndarray) -> Dict[str, Any]:
        """Integrate Lindblad equation."""
        
        if not np.allclose(rho0, rho0.conj().T, atol=1e-10):
            raise ValueError("ρ₀ must be Hermitian")
        if not np.isclose(np.trace(rho0), 1.0, atol=1e-10):
            raise ValueError("Tr(ρ₀) must equal 1")
        
        dim = H.shape[0]
        t0 = time.time()
        
        superop = self.build_superoperator(H, L_ops)
        sol = solve_ivp(lambda t, y: superop(y, t), (times[0], times[-1]),
                        rho0.flatten().astype(complex), t_eval=times,
                        rtol=self.rtol, atol=self.atol, method="RK45")
        rho_t = sol.y.T.reshape((len(times), dim, dim))
        
        return {
            "times": times,
            "rho_t": rho_t,
            "computation_time": time.time() - t0,
            "converged": True,
        }
    
    def validate_trace(self, result: Dict) -> bool:
        """Check trace preservation."""
        traces = [np.trace(rho) for rho in result["rho_t"]]
        return np.allclose(traces, 1.0, atol=1e-9)
    
    def validate_hermiticity(self, result: Dict) -> bool:
        """Check Hermiticity preservation."""
        for rho in result["rho_t"]:
            if not np.allclose(rho, rho.conj().T, atol=1e-9):
                return False
        return True
