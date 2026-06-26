"""Project I/O and serialization."""
import numpy as np
import json
import pickle
from typing import Dict, Any

class QuantumProject:
    """Project container for results and metadata."""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
        self.metadata = {}
        self.parameters = {}
    
    def add_result(self, key: str, result: Dict[str, Any]):
        """Store result."""
        self.results[key] = result
    
    def save(self, filename: str):
        """Save project to disk."""
        data = {
            "name": self.name,
            "metadata": self.metadata,
            "parameters": self.parameters,
            "results_keys": list(self.results.keys()),
        }
        
        # Save metadata as JSON
        with open(filename + ".meta.json", "w") as f:
            json.dump(data, f, indent=2)
        
        # Save results as pickle
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self.results, f)
    
    def load(self, filename: str):
        """Load project from disk."""
        with open(filename + ".meta.json", "r") as f:
            data = json.load(f)
        
        with open(filename + ".pkl", "rb") as f:
            self.results = pickle.load(f)
        
        self.name = data["name"]
        self.metadata = data["metadata"]
        self.parameters = data["parameters"]
