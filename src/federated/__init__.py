"""
src/federated/
 
Federated simulation layer for FedGAPrompt.
 
Milestone 6A: single-machine simulation (no Flower).
Milestone 6B: Flower replaces the communication loop in server.py.
"""
 
from .client import FederatedClient
from .server import FederatedServer
from .strategy import aggregate_best_prompt
 
__all__ = [
    "FederatedClient",
    "FederatedServer",
    "aggregate_best_prompt",
]
 