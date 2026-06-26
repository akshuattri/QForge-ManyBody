"""Comparison with QuTiP reference."""
import numpy as np

def compare_with_qutip(our_result: Dict[str, Any], 
                       qutip_result: Dict[str, Any]) -> Dict[str, float]:
    """Compare Lindblad results with QuTiP."""
    
    rho_ours = our_result["rho_t"]
    rho_qutip = np.array(qutip_result["rho_t"])
    
    # Compute errors
    errors = []
    for i in range(len(rho_ours)):
        err = np.linalg.norm(rho_ours[i] - rho_qutip[i], 'fro')
        errors.append(err)
    
    errors = np.array(errors)
    
    return {
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "final_error": float(errors[-1]),
        "agreement": bool(np.max(errors) < 1e-6),
    }
