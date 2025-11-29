"""
PALADIM Configuration
=====================
Centralized configuration for all PALADIM components.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layer."""
    num_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 768
    intermediate_dim: int = 3072
    dropout: float = 0.1
    load_balance_weight: float = 0.01  # Auxiliary loss coefficient
    capacity_factor: float = 1.25
    

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Plastic Memory) adapters."""
    rank: int = 16
    alpha: int = 32  # Scaling factor (typically 2 * rank)
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "query", "value", "key", "dense"
    ])
    

@dataclass
class ConsolidationConfig:
    """Configuration for the Consolidation Engine."""
    ewc_lambda: float = 5000.0  # EWC regularization strength
    fisher_samples: int = 200  # Samples for Fisher computation
    kd_temperature: float = 2.0  # Knowledge distillation temperature
    kd_alpha: float = 0.5  # KD loss weight
    consolidation_lr: float = 1e-5  # Learning rate during consolidation
    consolidation_steps: int = 100  # Steps per consolidation
    plastic_reset_ratio: float = 0.1  # How much LoRA to retain after consolidation


@dataclass
class MetaControllerConfig:
    """Configuration for the Meta-Controller."""
    consolidation_interval: int = 500  # Batches between consolidation checks
    novelty_threshold: float = 0.8  # Novelty score to trigger consolidation
    loss_plateau_patience: int = 50  # Steps of no improvement before consolidation
    loss_plateau_threshold: float = 0.005  # Minimum improvement to count
    expert_load_threshold: float = 0.95  # Expert load triggering spawn


@dataclass
class PALADIMConfig:
    """
    Complete PALADIM configuration.
    
    Pre Adaptive Learning Architecture of Dual-Process Hebbian-MoE Schema
    """
    # Base model
    model_name: str = "roberta-base"
    hidden_dim: int = 768
    num_labels: int = 2  # Default for binary classification
    
    # Component configs
    moe: MoEConfig = field(default_factory=MoEConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    meta_controller: MetaControllerConfig = field(default_factory=MetaControllerConfig)
    
    # Training
    plastic_lr: float = 5e-4  # Learning rate for LoRA adapters
    batch_size: int = 16
    max_seq_length: int = 512
    
    # Replay buffer
    buffer_capacity: int = 10000
    buffer_alpha: float = 0.6  # Priority exponent
    buffer_beta_start: float = 0.4  # IS weight exponent
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Update nested configs with parent values."""
        self.moe.hidden_dim = self.hidden_dim
        
    @classmethod
    def from_model_name(cls, model_name: str) -> "PALADIMConfig":
        """Create config based on model architecture."""
        if "roberta-base" in model_name or "bert-base" in model_name:
            return cls(
                model_name=model_name,
                hidden_dim=768,
                moe=MoEConfig(hidden_dim=768, intermediate_dim=3072),
            )
        elif "roberta-large" in model_name or "bert-large" in model_name:
            return cls(
                model_name=model_name,
                hidden_dim=1024,
                moe=MoEConfig(hidden_dim=1024, intermediate_dim=4096),
            )
        elif "mistral" in model_name.lower() or "llama" in model_name.lower():
            return cls(
                model_name=model_name,
                hidden_dim=4096,
                moe=MoEConfig(hidden_dim=4096, intermediate_dim=14336, num_experts=8),
                lora=LoRAConfig(rank=32, alpha=64),
            )
        else:
            return cls(model_name=model_name)

