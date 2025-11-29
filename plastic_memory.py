"""
PALADIM Plastic Memory (LoRA Adapters)
======================================
Task 2: Rapid adaptation layer using Low-Rank Adaptation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)
from peft.tuners.lora import LoraLayer


class PlasticMemory:
    """
    Plastic Memory Manager using LoRA adapters.
    
    Implements the fast, Hebbian-like adaptation pathway in PALADIM.
    The LoRA adapters (θ_P) are the only trainable parameters during
    rapid learning phases, while the base model (θ_C) remains frozen.
    
    Key responsibilities:
    - Apply LoRA adapters to target modules
    - Manage adapter training state
    - Partial reset after consolidation
    - Track adapter statistics
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        task_type: TaskType = TaskType.SEQ_CLS,
    ):
        """
        Initialize Plastic Memory with LoRA configuration.
        
        Args:
            model: Base model to adapt (Stable Core)
            rank: LoRA rank (r) - lower = less parameters, higher = more capacity
            alpha: LoRA alpha for scaling - typically 2*rank
            dropout: Dropout for LoRA layers
            target_modules: Which modules to apply LoRA to
            task_type: PEFT task type for proper configuration
        """
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.task_type = task_type
        
        # Default target modules for common architectures
        if target_modules is None:
            target_modules = self._get_default_targets(model)
        
        self.target_modules = target_modules
        
        # Create LoRA configuration
        self.lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=task_type,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(model, self.lora_config)
        
        # Freeze base model weights (Stable Core)
        self._freeze_base_model()
        
        # Store initial adapter state for reset
        self._store_initial_state()
        
    def _get_default_targets(self, model: nn.Module) -> List[str]:
        """Determine default LoRA target modules based on architecture."""
        # Check model type by examining module names
        module_names = [name for name, _ in model.named_modules()]
        
        if any('roberta' in name.lower() for name in module_names):
            return ["query", "value", "key", "dense"]
        elif any('bert' in name.lower() for name in module_names):
            return ["query", "value", "key", "dense"]
        elif any('llama' in name.lower() or 'mistral' in name.lower() 
                 for name in module_names):
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif any('gpt2' in name.lower() for name in module_names):
            return ["c_attn", "c_proj", "c_fc"]
        else:
            # Fallback: look for attention-like modules
            targets = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if any(key in name.lower() for key in ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']):
                        # Extract the attribute name
                        parts = name.split('.')
                        targets.append(parts[-1])
            return list(set(targets)) if targets else ["query", "value"]
    
    def _freeze_base_model(self):
        """Freeze all base model parameters, keeping only LoRA trainable."""
        for name, param in self.model.named_parameters():
            if 'lora_' not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    def _store_initial_state(self):
        """Store initial LoRA state for reset functionality."""
        self._initial_lora_state = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name.lower() and param.requires_grad:
                self._initial_lora_state[name] = param.data.clone()
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable (LoRA) parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def get_trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Get named trainable parameters."""
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
    
    def num_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def partial_reset(self, retain_ratio: float = 0.1):
        """
        Partially reset LoRA adapters after consolidation.
        
        Retains some of the learned patterns while clearing most
        to prepare for new learning.
        
        Args:
            retain_ratio: Fraction of weights to retain (0.0 = full reset)
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'lora_' in name.lower() and param.requires_grad:
                    if retain_ratio > 0:
                        # Scale down current weights
                        param.mul_(retain_ratio)
                    else:
                        # Full reset to initial state
                        if name in self._initial_lora_state:
                            param.copy_(self._initial_lora_state[name])
                        else:
                            param.zero_()
    
    def full_reset(self):
        """Fully reset LoRA adapters to initial state."""
        self.partial_reset(retain_ratio=0.0)
    
    def get_adapter_state(self) -> Dict[str, torch.Tensor]:
        """Get current state of all LoRA adapters."""
        state = {}
        for name, param in self.model.named_parameters():
            if 'lora_' in name.lower():
                state[name] = param.data.clone()
        return state
    
    def load_adapter_state(self, state: Dict[str, torch.Tensor]):
        """Load a saved adapter state."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.copy_(state[name])
    
    def get_adapter_statistics(self) -> Dict[str, float]:
        """Get statistics about adapter weights."""
        stats = {
            'total_params': 0,
            'mean_magnitude': 0.0,
            'max_magnitude': 0.0,
            'nonzero_ratio': 0.0,
        }
        
        all_params = []
        for name, param in self.model.named_parameters():
            if 'lora_' in name.lower() and param.requires_grad:
                flat = param.data.flatten()
                all_params.append(flat)
                stats['total_params'] += flat.numel()
        
        if all_params:
            all_params = torch.cat(all_params)
            stats['mean_magnitude'] = all_params.abs().mean().item()
            stats['max_magnitude'] = all_params.abs().max().item()
            stats['nonzero_ratio'] = (all_params != 0).float().mean().item()
        
        return stats
    
    def merge_and_unload(self) -> nn.Module:
        """
        Merge LoRA weights into base model and return unloaded model.
        
        This is typically done after consolidation when you want to
        "bake in" the learned adaptations.
        
        Returns:
            Base model with LoRA weights merged
        """
        return self.model.merge_and_unload()
    
    def forward(self, *args, **kwargs):
        """Forward pass through the adapted model."""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Allow direct calling of the plastic memory."""
        return self.forward(*args, **kwargs)
    
    @property
    def base_model(self) -> nn.Module:
        """Access the underlying base model."""
        return self.model.get_base_model()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self


def create_plastic_memory(
    model: nn.Module,
    config,
) -> PlasticMemory:
    """
    Factory function to create PlasticMemory from config.
    
    Args:
        model: Base model to adapt
        config: LoRAConfig with adapter parameters
        
    Returns:
        Configured PlasticMemory instance
    """
    return PlasticMemory(
        model=model,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        target_modules=config.target_modules,
    )

