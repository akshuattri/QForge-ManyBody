"""Base Solver class."""
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    """Abstract solver interface."""
    def __init__(self, rtol: float = 1e-8, atol: float = 1e-10, 
                 verbose: bool = False):
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose
        self.last_result = None
    
    @abstractmethod
    def solve(self, hamiltonian) -> Dict[str, Any]:
        pass
    
    def get_result(self) -> Dict[str, Any]:
        return self.last_result
