"""Complete test suite for QForge."""

import numpy as np
import pytest
from qforge.models.heisenberg import build_heisenberg_hamiltonian
from qforge.solvers.exact_diag import exact_diagonalization
from qforge.open_systems.lindblad import LindladSolver
from qforge.analysis.entropy import von_neumann_entropy, purity
from qforge.validation.analytical_solutions import validate_against_analytical

class TestExactDiagonalization:
    """ED solver tests."""
    
    def test_2spin_heisenberg(self):
        H = build_heisenberg_hamiltonian(2, J_x=1, J_y=1, J_z=1, h=0)
        result = exact_diagonalization(H)
        
        analytical = np.array([-0.75, -0.75, 0.25, 0.25])
        validation = validate_against_analytical(result["eigenvalues"], analytical)
        
        assert validation["agreement"]
        assert validation["max_abs_error"] < 1e-12
    
    def test_3spin_heisenberg(self):
        H = build_heisenberg_hamiltonian(3, J_x=1, J_y=1, J_z=1, h=0)
        result = exact_diagonalization(H)
        
        analytical = np.array([-1.75, -0.75, -0.75, 0.25, 0.25, 0.75, 1.25, 1.75])
        validation = validate_against_analytical(result["eigenvalues"], analytical)
        
        assert validation["agreement"]

class TestLindblad:
    """Lindblad solver tests."""
    
    def test_trace_preservation(self):
        H = np.zeros((2, 2), dtype=complex)
        L = np.array([[0, 0], [1, 0]], dtype=complex)
        rho0 = np.eye(2) / 2
        times = np.linspace(0, 1, 11)
        
        solver = LindladSolver()
        result = solver.integrate(H, rho0, [L], times)
        
        assert solver.validate_trace(result)
    
    def test_hermiticity_preservation(self):
        H = np.diag([1, -1])
        rho0 = np.eye(2) / 2
        times = np.linspace(0, 1, 11)
        
        solver = LindladSolver()
        result = solver.integrate(H, rho0, [], times)
        
        assert solver.validate_hermiticity(result)
    
    def test_decay_dynamics(self):
        H = np.zeros((2, 2), dtype=complex)
        decay = np.sqrt(1.0) * np.array([[0, 0], [1, 0]], dtype=complex)
        rho0 = np.array([[0, 0], [0, 1]], dtype=complex)
        
        times = np.linspace(0, 5, 51)
        
        solver = LindladSolver()
        result = solver.integrate(H, rho0, [decay], times)
        
        rho_final = result["rho_t"][-1]
        ground = np.array([[1, 0], [0, 0]], dtype=complex)
        
        assert np.allclose(rho_final, ground, atol=1e-6)

class TestAnalysis:
    """Analysis tools tests."""
    
    def test_entropy_pure_state(self):
        rho_pure = np.array([[1, 0], [0, 0]], dtype=complex)
        S = von_neumann_entropy(rho_pure)
        assert np.isclose(S, 0.0, atol=1e-12)
    
    def test_entropy_mixed_state(self):
        rho_mixed = np.eye(2, dtype=complex) / 2
        S = von_neumann_entropy(rho_mixed)
        assert np.isclose(S, np.log(2), atol=1e-12)
    
    def test_purity(self):
        rho = np.eye(3, dtype=complex) / 3
        P = purity(rho)
        assert np.isclose(P, 1.0/3, atol=1e-12)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
