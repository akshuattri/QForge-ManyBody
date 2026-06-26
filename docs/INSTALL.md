# QForge Installation Guide

## Requirements
- Python 3.10+
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.5

## Installation

### From Source
```bash
git clone https://github.com/akshuattri/QForge.git
cd QForge
pip install -e .
```

### With Optional Dependencies
```bash
pip install -e ".[qutip]"  # For QuTiP comparison
pip install -e ".[dev]"    # For development
```

## Verification
```bash
python -c "import qforge; print('QForge installed successfully')"
pytest validation/ -v
```

## Quick Test
```python
from qforge.models.heisenberg import build_heisenberg_hamiltonian
from qforge.solvers.exact_diag import exact_diagonalization

H = build_heisenberg_hamiltonian(4, J_x=1, J_y=1, J_z=1)
result = exact_diagonalization(H, verbose=True)
print(f"Ground state energy: {result['eigenvalues'][0]:.8f}")
```
