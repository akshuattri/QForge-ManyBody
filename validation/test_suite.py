
"""Complete validation suite with benchmarks."""

import numpy as np
import pytest

def test_exact_diag_2spin():
    """Test 2-spin Heisenberg XXX."""
    from qforge.models.heisenberg import build_heisenberg_hamiltonian
    from qforge.solvers.exact_diag import exact_diagonalization
    
    H = build_heisenberg_hamiltonian(2, J_x=1, J_y=1, J_z=1, h=0)
    result = exact_diagonalization(H)
    
    # Analytical: -3/4, -3/4, 1/4, 1/4
    analytical = np.array([-0.75, -0.75, 0.25, 0.25])
    assert np.allclose(np.sort(result["eigenvalues"]), analytical, atol=1e-12)

def test_lindblad_decay():
    """Test Lindblad decay to ground state."""
    from qforge.solvers.lindblad import integrate_lindblad
    
    H = np.zeros((2, 2), dtype=complex)
    L = np.sqrt(1.0) * np.array([[0, 0], [1, 0]], dtype=complex)
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)
    
    times = np.linspace(0, 3, 31)
    result = integrate_lindblad(H, rho0, [L], times)
    
    # Final state should be ground state
    rho_final = result["rho_t"][-1]
    ground = np.array([[1, 0], [0, 0]], dtype=complex)
    assert np.allclose(rho_final, ground, atol=1e-7)

def test_trace_preservation():
    """Test Lindblad preserves trace."""
    from qforge.solvers.lindblad import integrate_lindblad
    
    dim = 3
    H = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    H = (H + H.conj().T)/2
    rho0 = np.eye(dim)/dim
    L = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    
    times = np.linspace(0, 1, 11)
    result = integrate_lindblad(H, rho0, [L], times)
    
    for rho in result["rho_t"]:
        trace = np.trace(rho)
        assert np.isclose(trace, 1.0, atol=1e-10)

def test_hermiticity():
    """Test Lindblad preserves Hermiticity."""
    from qforge.solvers.lindblad import integrate_lindblad
    
    H = np.diag([0, 1, 2])
    rho0 = np.eye(3)/3
    
    times = np.linspace(0, 1, 11)
    result = integrate_lindblad(H, rho0, [], times)
    
    for rho in result["rho_t"]:
        assert np.allclose(rho, rho.conj().T, atol=1e-12)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
