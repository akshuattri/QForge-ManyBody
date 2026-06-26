"""XXZ Heisenberg model."""
import numpy as np
from qforge.physics.spin import spin_operators

def build_xxz(N: int, Jx: float, Jy: float, Jz: float, h: float = 0,
              bc: str = "open") -> np.ndarray:
    """XXZ Heisenberg: H = Σ_i [Jx Sx + Jy Sy + Jz Sz] + h Σ_i Sz"""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    sx, sy, sz, _, _ = spin_operators(0.5)
    I = np.eye(2, dtype=complex)
    
    def kron_op(ops):
        r = ops[0]
        for o in ops[1:]:
            r = np.kron(r, o)
        return r
    
    n_bonds = N if bc == "periodic" else N-1
    for i in range(n_bonds):
        j = (i+1) % N
        for k in range(N):
            if k == i:
                op_x = sx
                op_y = sy
                op_z = sz
            elif k == j:
                op_x = sx
                op_y = sy
                op_z = sz
            else:
                op_x = op_y = op_z = I
        if Jx != 0:
            ops_x = [I]*N
            ops_x[i], ops_x[j] = sx, sx
            H += Jx * kron_op(ops_x)
        if Jy != 0:
            ops_y = [I]*N
            ops_y[i], ops_y[j] = sy, sy
            H += Jy * kron_op(ops_y)
        if Jz != 0:
            ops_z = [I]*N
            ops_z[i], ops_z[j] = sz, sz
            H += Jz * kron_op(ops_z)
    
    if h != 0:
        for i in range(N):
            ops = [I]*N
            ops[i] = sz
            H += h * kron_op(ops)
    
    return (H + H.conj().T) / 2
