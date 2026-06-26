"""Quantum trajectories (MCWF method)."""
import numpy as np

def quantum_trajectory_single(H: np.ndarray, L_ops: List[np.ndarray],
                              psi0: np.ndarray, times: np.ndarray,
                              dt: float = None) -> Dict[str, Any]:
    """Single quantum trajectory."""
    
    if dt is None:
        dt = times[1] - times[0] if len(times) > 1 else 0.01
    
    psi = psi0.copy()
    trajectories = [psi.copy()]
    
    for t in times[1:]:
        # Effective non-Hermitian evolution
        H_eff = H.copy()
        for L in L_ops:
            H_eff -= 0.5j * L.conj().T @ L
        
        # Time step
        U = np.linalg.matrix_power(np.eye(len(psi)) - 1j*dt*H_eff, -1)
        psi = U @ psi
        
        # Renormalize
        psi /= np.linalg.norm(psi)
        
        # Jump with probability
        p_jump = 1 - np.abs(np.linalg.norm(psi))**2
        if np.random.rand() < p_jump:
            # Apply random jump operator
            idx = np.random.randint(len(L_ops))
            psi = L_ops[idx] @ psi
            psi /= np.linalg.norm(psi)
        
        trajectories.append(psi.copy())
    
    return {
        "times": times,
        "psi_t": np.array(trajectories),
    }
