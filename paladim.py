"""
PALADIM Main System
===================
Pre Adaptive Learning Architecture of Dual-Process Hebbian-MoE Schema

Complete integration of all PALADIM components into a unified
continual learning system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Iterator, Tuple, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass

from .config import PALADIMConfig
from .moe_layer import MoELayer, replace_ffn_with_moe, get_moe_layers, get_total_aux_loss
from .plastic_memory import PlasticMemory, create_plastic_memory
from .consolidation import ConsolidationEngine
from .meta_controller import MetaController


@dataclass
class PALADIMOutput:
    """Output container for PALADIM forward pass."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    router_probs: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None


@dataclass
class TrainingResult:
    """Result of a training step."""
    loss: float
    consolidated: bool = False
    expert_spawned: bool = False
    metrics: Optional[Dict] = None


class PALADIM(nn.Module):
    """
    PALADIM: Pre Adaptive Learning Architecture of Dual-Process Hebbian-MoE Schema
    
    A continual learning system that combines:
    - Stable Core (θ_C): Pre-trained Transformer base weights (frozen during rapid learning)
    - Plastic Memory (θ_P): LoRA adapters for rapid Hebbian-like adaptation
    - MoE Layer: Sparse capacity scaling with dynamic expert spawning
    - Consolidation Engine: EWC + KD for protected knowledge transfer
    - Meta-Controller: Adaptive consolidation timing
    
    Training Phases:
    1. Rapid Learning: Only LoRA adapters (θ_P) are updated
    2. Consolidation: Knowledge transfers from θ_P to θ_C with protection
    
    Example:
        >>> config = PALADIMConfig(model_name="roberta-base")
        >>> model = PALADIM(config)
        >>> 
        >>> # Training loop
        >>> for batch in data_stream:
        ...     result = model.step(batch)
        ...     if result.consolidated:
        ...         print("Consolidation occurred")
    """
    
    def __init__(self, config: PALADIMConfig):
        super().__init__()
        self.config = config
        
        # Load base model (Stable Core θ_C)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
        )
        
        # Replace FFN with MoE layers
        self.base_model = replace_ffn_with_moe(
            self.base_model,
            config.moe,
        )
        
        # Apply LoRA adapters (Plastic Memory θ_P)
        self.plastic_memory = PlasticMemory(
            model=self.base_model,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )
        
        # The model with LoRA applied
        self.model = self.plastic_memory.model
        
        # Initialize Consolidation Engine
        self.consolidation_engine = ConsolidationEngine(
            ewc_lambda=config.consolidation.ewc_lambda,
            kd_temperature=config.consolidation.kd_temperature,
            kd_alpha=config.consolidation.kd_alpha,
            consolidation_lr=config.consolidation.consolidation_lr,
            consolidation_steps=config.consolidation.consolidation_steps,
            plastic_reset_ratio=config.consolidation.plastic_reset_ratio,
            fisher_samples=config.consolidation.fisher_samples,
        )
        
        # Initialize Meta-Controller
        self.meta_controller = MetaController(
            embedding_dim=config.hidden_dim,
            consolidation_interval=config.meta_controller.consolidation_interval,
            novelty_threshold=config.meta_controller.novelty_threshold,
            loss_plateau_patience=config.meta_controller.loss_plateau_patience,
            loss_plateau_threshold=config.meta_controller.loss_plateau_threshold,
            expert_load_threshold=config.meta_controller.expert_load_threshold,
        )
        
        # Training state
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.current_task_id: Optional[str] = None
        self.task_history: List[str] = []
        
        # Move to device
        self.to(config.device)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PALADIMOutput:
        """
        Forward pass through PALADIM.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels for computing loss
            
        Returns:
            PALADIMOutput containing loss, logits, and auxiliary information
        """
        # Forward through the LoRA-adapted model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )
        
        # Get MoE auxiliary losses
        aux_loss = get_total_aux_loss(self.model)
        
        # Combine losses
        total_loss = None
        if outputs.loss is not None:
            total_loss = outputs.loss + aux_loss
        
        # Get router probabilities from MoE layers
        router_probs = None
        moe_layers = get_moe_layers(self.model)
        if moe_layers:
            # Just return from first layer for simplicity
            # In practice, you might want to aggregate these
            router_probs = moe_layers[0].get_expert_utilization()
        
        return PALADIMOutput(
            loss=total_loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states[-1] if outputs.hidden_states else None,
            router_probs=router_probs,
            aux_loss=aux_loss,
        )
    
    def learn_task(
        self,
        dataloader: Iterator,
        num_epochs: int = 1,
        learning_rate: Optional[float] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Rapid Learning Phase: Train only LoRA adapters (θ_P).
        
        The Stable Core (θ_C) remains frozen. Only the Plastic Memory
        (LoRA adapters) receives gradient updates.
        
        Args:
            dataloader: Training data iterator
            num_epochs: Number of training epochs
            learning_rate: Learning rate (uses config default if None)
            task_id: Optional identifier for the task
            
        Returns:
            Training metrics dictionary
        """
        if task_id:
            self.current_task_id = task_id
            if task_id not in self.task_history:
                self.task_history.append(task_id)
        
        lr = learning_rate or self.config.plastic_lr
        
        # Create optimizer for LoRA parameters only
        if self.optimizer is None or learning_rate is not None:
            self.optimizer = torch.optim.AdamW(
                self.plastic_memory.get_trainable_parameters(),
                lr=lr,
                weight_decay=0.01,
            )
        
        self.train()
        
        total_loss = 0.0
        num_steps = 0
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                # Move to device
                batch = self._prepare_batch(batch)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(**batch)
                
                if outputs.loss is None:
                    raise ValueError("No loss computed. Ensure labels are provided.")
                
                # Backward pass
                outputs.loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.plastic_memory.get_trainable_parameters(),
                    1.0,
                )
                
                self.optimizer.step()
                
                total_loss += outputs.loss.item()
                num_steps += 1
                
                # Meta-controller step
                expert_loads = outputs.router_probs
                embeddings = outputs.hidden_states
                
                decisions = self.meta_controller.step(
                    loss=outputs.loss.item(),
                    embeddings=embeddings.mean(dim=1) if embeddings is not None else None,
                    expert_loads=expert_loads,
                )
                
                # Apply learning rate modulation
                if decisions['lr_multiplier'] != 1.0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr * decisions['lr_multiplier']
        
        return {
            'avg_loss': total_loss / num_steps if num_steps > 0 else 0,
            'num_steps': num_steps,
            'num_epochs': num_epochs,
            'task_id': task_id,
        }
    
    def consolidate_knowledge(
        self,
        dataloader: Iterator,
        validation_dataloaders: Optional[Dict[str, Iterator]] = None,
    ) -> Dict[str, Any]:
        """
        Consolidation Phase: Transfer knowledge from θ_P to θ_C.
        
        This phase:
        1. Computes Fisher Information on Stable Core
        2. Temporarily unfreezes Stable Core
        3. Trains with EWC + KD loss
        4. Re-freezes Stable Core
        5. Partially resets Plastic Memory
        
        Args:
            dataloader: Data for consolidation training
            validation_dataloaders: Optional dict of task -> dataloader for forgetting check
            
        Returns:
            Consolidation metrics
        """
        # Prepare consolidation (Fisher computation, teacher storage)
        self.consolidation_engine.prepare_consolidation(
            model=self.model,
            dataloader=dataloader,
        )
        
        # Define task loss function
        def task_loss_fn(outputs, batch):
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                return outputs.loss
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1),
            )
        
        # Run consolidation
        metrics = self.consolidation_engine.consolidate(
            model=self.model,
            plastic_memory=self.plastic_memory,
            dataloader=dataloader,
            task_loss_fn=task_loss_fn,
        )
        
        # Notify meta-controller
        self.meta_controller.on_consolidation_complete()
        
        # Validate for forgetting
        if validation_dataloaders:
            def metric_fn(outputs, batch):
                preds = outputs.logits.argmax(dim=-1)
                labels = batch['labels']
                return (preds == labels).float().mean().item()
            
            forgetting_check = self.consolidation_engine.validate_forgetting(
                model=self.model,
                validation_dataloaders=validation_dataloaders,
                metric_fn=metric_fn,
            )
            metrics['forgetting_check'] = forgetting_check
        
        return metrics
    
    def step(
        self,
        batch: Dict[str, torch.Tensor],
        domain_id: Optional[str] = None,
    ) -> TrainingResult:
        """
        Single training step with automatic consolidation management.
        
        Combines rapid learning with meta-controller decisions for
        automatic consolidation triggering.
        
        Args:
            batch: Training batch
            domain_id: Optional domain/task identifier
            
        Returns:
            TrainingResult with loss and status flags
        """
        self.train()
        
        batch = self._prepare_batch(batch)
        
        # Create optimizer if needed
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.plastic_memory.get_trainable_parameters(),
                lr=self.config.plastic_lr,
                weight_decay=0.01,
            )
        
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.forward(**batch)
        
        if outputs.loss is None:
            raise ValueError("No loss computed. Ensure labels are provided.")
        
        # Backward and update
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.plastic_memory.get_trainable_parameters(),
            1.0,
        )
        self.optimizer.step()
        
        # Meta-controller decision
        decisions = self.meta_controller.step(
            loss=outputs.loss.item(),
            embeddings=outputs.hidden_states.mean(dim=1) if outputs.hidden_states is not None else None,
            expert_loads=outputs.router_probs,
        )
        
        consolidated = False
        expert_spawned = False
        
        # Handle consolidation trigger
        if decisions['consolidate']:
            # Would need a dataloader for proper consolidation
            # This is a simplified trigger notification
            consolidated = True
        
        # Handle expert spawning
        if decisions['spawn_expert']:
            moe_layers = get_moe_layers(self.model)
            for moe in moe_layers:
                if moe.num_experts < 32:  # Max experts limit
                    moe.spawn_expert()
                    expert_spawned = True
                    break
        
        return TrainingResult(
            loss=outputs.loss.item(),
            consolidated=consolidated,
            expert_spawned=expert_spawned,
            metrics=decisions,
        )
    
    def _prepare_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        device = next(self.parameters()).device
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        plastic = self.plastic_memory.num_trainable_parameters()
        
        return {
            'total': total,
            'trainable': trainable,
            'plastic_memory': plastic,
            'stable_core': total - plastic,
            'trainable_ratio': trainable / total if total > 0 else 0,
        }
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get MoE expert statistics."""
        moe_layers = get_moe_layers(self.model)
        
        if not moe_layers:
            return {'num_moe_layers': 0}
        
        stats = {
            'num_moe_layers': len(moe_layers),
            'experts_per_layer': [moe.num_experts for moe in moe_layers],
            'expert_loads': [moe.get_expert_utilization().tolist() for moe in moe_layers],
        }
        
        return stats
    
    def save_pretrained(self, path: str):
        """Save model to disk."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapters
        self.plastic_memory.model.save_pretrained(path)
        
        # Save config
        torch.save({
            'config': self.config,
            'meta_controller_state': self.meta_controller.get_statistics(),
            'consolidation_count': self.consolidation_engine.consolidation_count,
            'task_history': self.task_history,
        }, f"{path}/paladim_state.pt")
    
    @classmethod
    def from_pretrained(cls, path: str) -> "PALADIM":
        """Load model from disk."""
        state = torch.load(f"{path}/paladim_state.pt")
        config = state['config']
        
        model = cls(config)
        
        # Load LoRA weights
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.base_model, path)
        model.plastic_memory.model = model.model
        
        model.task_history = state.get('task_history', [])
        
        return model

