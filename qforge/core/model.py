"""Base Model class."""
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelParameters:
    """Parameter container."""
    system_size: int
    parameters: Dict[str, float]
    metadata: Dict[str, Any]

class BaseModel:
    """Abstract base for all quantum models."""
    def __init__(self, system_size: int):
        self.system_size = system_size
        self.parameters = {}
        
    def set_parameter(self, name: str, value: float):
        self.parameters[name] = float(value)
    
    def build_hamiltonian(self):
        raise NotImplementedError
    
    def get_hilbert_dimension(self) -> int:
        raise NotImplementedError
