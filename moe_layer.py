"""
PALADIM Mixture of Experts Layer
================================
Task 1: Sparse MoE implementation with top-k gating and load balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import copy


class Expert(nn.Module):
    """
    Single Expert Network (FFN).
    
    Standard Transformer FFN: Linear -> GELU -> Linear
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
               or [num_tokens, hidden_dim] (flattened)
        Returns:
            Output tensor of same shape as input
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GatingNetwork(nn.Module):
    """
    Router/Gating network for MoE.
    
    Computes routing probabilities for each expert.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.noise_std = noise_std
        self.num_experts = num_experts
        
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Compute routing logits.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim] or [num_tokens, hidden_dim]
            training: Whether to add noise for exploration
            
        Returns:
            Router logits [batch_size, seq_len, num_experts] or [num_tokens, num_experts]
        """
        logits = self.gate(x)
        
        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
            
        return logits


class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts Layer with Top-K Routing.
    
    Replaces the standard FFN in Transformer blocks.
    Implements:
    - Top-k expert selection
    - Weighted combination of expert outputs
    - Load balancing auxiliary loss
    - Capacity factor for efficiency
    
    Architecture:
        Input -> Gating Network -> Top-K Selection -> Experts -> Weighted Sum -> Output
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.capacity_factor = capacity_factor
        
        # Gating network (router)
        self.gate = GatingNetwork(hidden_dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(hidden_dim, intermediate_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Track expert loads for monitoring
        self.register_buffer('expert_load', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        
        # Store auxiliary loss for training
        self.aux_loss = torch.tensor(0.0)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional mask [batch_size, seq_len]
            
        Returns:
            Tuple of:
                - Output tensor [batch_size, seq_len, hidden_dim]
                - Router probabilities [batch_size, seq_len, num_experts]
        """
        original_shape = x.shape
        batch_size, seq_len, hidden_dim = x.shape
        
        # Flatten for routing: [batch_size * seq_len, hidden_dim]
        x_flat = x.view(-1, hidden_dim)
        num_tokens = x_flat.shape[0]
        
        # Get routing logits and probabilities
        router_logits = self.gate(x_flat, training=self.training)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # [num_tokens]
            expert_weights = top_k_probs[:, k:k+1]  # [num_tokens, 1]
            
            for expert_idx in range(self.num_experts):
                # Find tokens routed to this expert
                mask = (expert_indices == expert_idx)
                
                if mask.any():
                    # Get tokens for this expert
                    expert_input = x_flat[mask]
                    
                    # Compute expert output
                    expert_output = self.experts[expert_idx](expert_input)
                    
                    # Weight and accumulate
                    output[mask] += expert_weights[mask] * expert_output
        
        # Compute load balancing auxiliary loss
        if self.training:
            self.aux_loss = self._compute_load_balance_loss(router_probs)
            self._update_expert_load(router_probs)
        
        # Reshape back to original shape
        output = output.view(original_shape)
        router_probs = router_probs.view(batch_size, seq_len, self.num_experts)
        
        return output, router_probs
    
    def _compute_load_balance_loss(
        self,
        router_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        Encourages even distribution of tokens across experts.
        
        L_aux = α * N * Σ(f_i * P_i)
        where:
            f_i = fraction of tokens routed to expert i
            P_i = average routing probability for expert i
        """
        # Fraction of tokens routed to each expert (based on top-1)
        top1_indices = router_probs.argmax(dim=-1)  # [num_tokens]
        
        # Count tokens per expert
        tokens_per_expert = torch.zeros(
            self.num_experts, device=router_probs.device
        )
        for i in range(self.num_experts):
            tokens_per_expert[i] = (top1_indices == i).float().sum()
        
        # Fraction per expert
        f = tokens_per_expert / (router_probs.shape[0] + 1e-9)
        
        # Average probability per expert
        P = router_probs.mean(dim=0)
        
        # Load balance loss
        aux_loss = self.load_balance_weight * self.num_experts * (f * P).sum()
        
        return aux_loss
    
    def _update_expert_load(self, router_probs: torch.Tensor):
        """Update running average of expert loads."""
        with torch.no_grad():
            current_load = router_probs.mean(dim=0)
            # Exponential moving average
            self.expert_load = 0.95 * self.expert_load + 0.05 * current_load
            self.total_tokens += router_probs.shape[0]
    
    def get_expert_utilization(self) -> torch.Tensor:
        """Get current expert utilization statistics."""
        return self.expert_load.clone()
    
    def spawn_expert(self, template_idx: int = 0) -> int:
        """
        Spawn a new expert by cloning an existing one.
        
        Args:
            template_idx: Index of expert to clone
            
        Returns:
            Index of the new expert
        """
        # Clone the template expert
        new_expert = copy.deepcopy(self.experts[template_idx])
        
        # Add noise for differentiation
        with torch.no_grad():
            for param in new_expert.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        
        # Add to expert list
        self.experts.append(new_expert)
        
        # Expand gating network
        old_weight = self.gate.gate.weight.data
        new_weight = torch.zeros(
            self.num_experts + 1,
            self.hidden_dim,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        new_weight[:self.num_experts] = old_weight
        new_weight[self.num_experts] = old_weight[template_idx] + \
            torch.randn_like(old_weight[template_idx]) * 0.01
        
        self.gate.gate = nn.Linear(
            self.hidden_dim,
            self.num_experts + 1,
            bias=False,
            device=old_weight.device,
        )
        self.gate.gate.weight.data = new_weight
        self.gate.num_experts = self.num_experts + 1
        
        # Update expert load buffer
        new_load = torch.zeros(
            self.num_experts + 1,
            device=self.expert_load.device,
        )
        new_load[:self.num_experts] = self.expert_load
        self.expert_load = new_load
        
        self.num_experts += 1
        
        return self.num_experts - 1


def replace_ffn_with_moe(
    model: nn.Module,
    config,
    layer_indices: Optional[list] = None,
) -> nn.Module:
    """
    Replace FFN layers in a Transformer model with MoE layers.
    
    Args:
        model: The base Transformer model
        config: MoEConfig with MoE parameters
        layer_indices: Which layers to replace (None = all)
        
    Returns:
        Modified model with MoE layers
    """
    # Detect model architecture
    if hasattr(model, 'roberta'):
        encoder = model.roberta.encoder
        layers = encoder.layer
    elif hasattr(model, 'bert'):
        encoder = model.bert.encoder
        layers = encoder.layer
    elif hasattr(model, 'model'):  # LLaMA/Mistral style
        layers = model.model.layers
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")
    
    if layer_indices is None:
        layer_indices = list(range(len(layers)))
    
    for idx in layer_indices:
        layer = layers[idx]
        
        # Get FFN dimensions from existing layer
        if hasattr(layer, 'intermediate'):  # BERT/RoBERTa
            intermediate_dim = layer.intermediate.dense.out_features
            hidden_dim = layer.intermediate.dense.in_features
            
            # Create MoE layer
            moe = MoELayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                dropout=config.dropout,
                load_balance_weight=config.load_balance_weight,
                capacity_factor=config.capacity_factor,
            )
            
            # Store original FFN for potential KD
            layer._original_ffn = nn.Sequential(
                layer.intermediate,
                layer.output,
            )
            
            # Replace with MoE
            layer.moe = moe
            layer._use_moe = True
            
        elif hasattr(layer, 'mlp'):  # LLaMA/Mistral style
            mlp = layer.mlp
            if hasattr(mlp, 'gate_proj'):
                intermediate_dim = mlp.gate_proj.out_features
                hidden_dim = mlp.gate_proj.in_features
            else:
                intermediate_dim = mlp.fc1.out_features
                hidden_dim = mlp.fc1.in_features
            
            moe = MoELayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                dropout=config.dropout,
                load_balance_weight=config.load_balance_weight,
            )
            
            layer._original_mlp = mlp
            layer.moe = moe
            layer._use_moe = True
    
    return model


def get_moe_layers(model: nn.Module) -> list:
    """Get all MoE layers from a model."""
    moe_layers = []
    for module in model.modules():
        if isinstance(module, MoELayer):
            moe_layers.append(module)
    return moe_layers


def get_total_aux_loss(model: nn.Module) -> torch.Tensor:
    """Aggregate auxiliary losses from all MoE layers."""
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for moe in get_moe_layers(model):
        total_loss = total_loss + moe.aux_loss
    return total_loss

