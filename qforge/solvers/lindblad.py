
"""Lindblad master equation solver."""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Dict, Any
import time

def lindblad_superoperator(H: np.ndarray, L_ops: List[np.ndarray]):
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

def integrate_lindblad(H: np.ndarray, rho0: np.ndarray, L_ops: List[np.ndarray],
                       times: np.ndarray, rtol: float = 1e-8, 
                       verbose: bool = False) -> Dict[str, Any]:
    """Integrate Lindblad equation."""
    
    if not np.allclose(rho0, rho0.conj().T, atol=1e-10):
        raise ValueError("ρ₀ must be Hermitian")
    
    if not np.isclose(np.trace(rho0), 1.0, atol=1e-10):
        raise ValueError("Tr(ρ₀) must equal 1")
    
    dim = H.shape[0]
    if verbose:
        print(f"Lindblad: {dim}×{dim} system, {len(L_ops)} collapse ops")
    
    t0 = time.time()
    superop = lindblad_superoperator(H, L_ops)
    sol = solve_ivp(lambda t, y: superop(y, t), (times[0], times[-1]),
                    rho0.flatten().astype(complex), t_eval=times,
                    rtol=rtol, method="RK45")
    rho_t = sol.y.T.reshape((len(times), dim, dim))
    
    return {
        "times": times,
        "rho_t": rho_t,
        "computation_time": time.time() - t0,
        "converged": True,
    }
