# PALADIM: Pre Adaptive Learning Architecture of Dual-Process Hebbian-MoE Schema
# A Continual Learning System for NLP

from .config import PALADIMConfig
from .moe_layer import MoELayer, Expert
from .plastic_memory import PlasticMemory
from .consolidation import ConsolidationEngine, EWCLoss, KnowledgeDistillationLoss
from .meta_controller import MetaController
from .paladim import PALADIM

__version__ = "0.1.0"
__all__ = [
    "PALADIMConfig",
    "MoELayer",
    "Expert", 
    "PlasticMemory",
    "ConsolidationEngine",
    "EWCLoss",
    "KnowledgeDistillationLoss",
    "MetaController",
    "PALADIM",
]

