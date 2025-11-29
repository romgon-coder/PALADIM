"""
PALADIM Consolidation Engine
============================
Task 3: Knowledge transfer from Plastic Memory to Stable Core.

Implements:
- Elastic Weight Consolidation (EWC) loss
- Knowledge Distillation (KD) loss
- Composite consolidation loss
- Fisher Information Matrix computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Iterator
from collections import OrderedDict
import copy


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation Loss.
    
    Prevents catastrophic forgetting by penalizing changes to important weights.
    Importance is measured by the Fisher Information Matrix.
    
    L_ewc = Σ_i (λ/2) * F_i * (θ_i - θ*_i)²
    
    where:
        λ: Regularization strength
        F_i: Fisher Information for parameter i
        θ_i: Current parameter value
        θ*_i: Optimal parameter value (from previous consolidation)
    """
    
    def __init__(
        self,
        lambda_ewc: float = 5000.0,
    ):
        super().__init__()
        self.lambda_ewc = lambda_ewc
        
        # Storage for optimal weights and Fisher matrix
        # These are computed on Stable Core (θ_C) weights ONLY
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        
        # Track which parameters are protected
        self.protected_params: set = set()
        
    def register_optimal_params(
        self,
        model: nn.Module,
        param_filter: Optional[callable] = None,
    ):
        """
        Store current parameters as optimal θ*.
        
        Should be called after each successful consolidation or
        after training on a new task.
        
        Args:
            model: The model containing parameters to store
            param_filter: Optional function to filter parameters
                         Default: exclude LoRA parameters
        """
        if param_filter is None:
            param_filter = lambda name: 'lora_' not in name.lower()
        
        self.optimal_params.clear()
        
        for name, param in model.named_parameters():
            if param_filter(name) and param.requires_grad:
                self.optimal_params[name] = param.data.clone().detach()
                self.protected_params.add(name)
    
    def compute_fisher(
        self,
        model: nn.Module,
        dataloader: Iterator,
        num_samples: int = 200,
        param_filter: Optional[callable] = None,
    ):
        """
        Compute Fisher Information Matrix (diagonal approximation).
        
        Fisher is computed as the expected squared gradient:
        F_i = E[(∂L/∂θ_i)²]
        
        CRITICAL: Must be computed on Stable Core (θ_C) weights ONLY,
        not on Plastic Memory (LoRA) weights.
        
        Args:
            model: The model to compute Fisher for
            dataloader: Data iterator for sampling
            num_samples: Number of samples for estimation
            param_filter: Filter for which parameters to include
        """
        if param_filter is None:
            param_filter = lambda name: 'lora_' not in name.lower()
        
        model.eval()
        
        # Initialize Fisher
        self.fisher.clear()
        for name, param in model.named_parameters():
            if param_filter(name) and param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)
        
        sample_count = 0
        
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(next(model.parameters()).device) 
                        if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                input_ids = batch.get('input_ids')
                labels = batch.get('labels', input_ids)
            else:
                input_ids = batch[0].to(next(model.parameters()).device)
                labels = batch[1].to(next(model.parameters()).device) if len(batch) > 1 else input_ids
            
            model.zero_grad()
            
            # Forward pass
            outputs = model(input_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute log-likelihood
            if logits.dim() == 3:  # [batch, seq, vocab]
                log_probs = F.log_softmax(logits, dim=-1)
                # Use model predictions as pseudo-labels for Fisher
                predicted = logits.argmax(dim=-1)
                selected_log_probs = log_probs.gather(
                    -1, predicted.unsqueeze(-1)
                ).squeeze(-1)
                loss = -selected_log_probs.mean()
            else:  # [batch, num_classes]
                log_probs = F.log_softmax(logits, dim=-1)
                if labels.dim() == 0 or labels.shape == logits.shape[:-1]:
                    loss = F.cross_entropy(logits, labels)
                else:
                    loss = -log_probs.mean()
            
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if name in self.fisher and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
            
            sample_count += input_ids.shape[0]
        
        # Normalize
        for name in self.fisher:
            self.fisher[name] /= sample_count
            
        model.train()
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC loss.
        
        L_ewc = Σ_i (λ/2) * F_i * (θ_i - θ*_i)²
        
        Args:
            model: Model with current parameters
            
        Returns:
            EWC loss value
        """
        if not self.optimal_params or not self.fisher:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if name in self.optimal_params and name in self.fisher:
                optimal = self.optimal_params[name]
                fisher = self.fisher[name]
                
                loss += (fisher * (param - optimal) ** 2).sum()
        
        return (self.lambda_ewc / 2) * loss


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for Consolidation.
    
    Uses the previous state of the Stable Core as "teacher"
    to regularize updates during consolidation.
    
    L_kd = KL(softmax(z_teacher/T) || softmax(z_student/T)) * T²
    
    where T is the temperature for softening distributions.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
        # Store teacher model state
        self.teacher_state: Optional[Dict[str, torch.Tensor]] = None
        
    def register_teacher(self, model: nn.Module):
        """
        Store current model state as teacher.
        
        Called before consolidation to capture the pre-consolidation
        model as the knowledge distillation target.
        """
        self.teacher_state = OrderedDict()
        for name, param in model.named_parameters():
            self.teacher_state[name] = param.data.clone().detach()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KD loss between student and teacher outputs.
        
        Args:
            student_logits: Current model outputs
            teacher_logits: Teacher (previous state) outputs
            
        Returns:
            KD loss value
        """
        T = self.temperature
        
        # Softmax with temperature
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        
        # KL divergence (scaled by T²)
        kd_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean',
        ) * (T ** 2)
        
        return self.alpha * kd_loss


class ConsolidationEngine:
    """
    Consolidation Engine for PALADIM.
    
    Orchestrates the knowledge transfer from Plastic Memory (θ_P)
    to Stable Core (θ_C) while preventing catastrophic forgetting.
    
    Consolidation Process:
    1. Compute Fisher Information on Stable Core weights
    2. Store current Stable Core as teacher for KD
    3. Unfreeze Stable Core for updates
    4. Train with composite loss: L_task + L_ewc + L_kd
    5. Re-freeze Stable Core
    6. Partially reset Plastic Memory
    7. Update optimal weights for next EWC
    """
    
    def __init__(
        self,
        ewc_lambda: float = 5000.0,
        kd_temperature: float = 2.0,
        kd_alpha: float = 0.5,
        consolidation_lr: float = 1e-5,
        consolidation_steps: int = 100,
        plastic_reset_ratio: float = 0.1,
        fisher_samples: int = 200,
    ):
        self.ewc_loss = EWCLoss(lambda_ewc=ewc_lambda)
        self.kd_loss = KnowledgeDistillationLoss(
            temperature=kd_temperature,
            alpha=kd_alpha,
        )
        
        self.consolidation_lr = consolidation_lr
        self.consolidation_steps = consolidation_steps
        self.plastic_reset_ratio = plastic_reset_ratio
        self.fisher_samples = fisher_samples
        
        # Track consolidation history
        self.consolidation_count = 0
        self.consolidation_losses: list = []
        
        # Teacher model for KD
        self.teacher_model: Optional[nn.Module] = None
    
    def prepare_consolidation(
        self,
        model: nn.Module,
        dataloader: Iterator,
    ):
        """
        Prepare for consolidation phase.
        
        1. Compute Fisher Information on Stable Core
        2. Store current model as teacher
        
        Args:
            model: The full PALADIM model
            dataloader: Data for Fisher computation
        """
        # Compute Fisher on Stable Core (excluding LoRA)
        self.ewc_loss.compute_fisher(
            model=model,
            dataloader=dataloader,
            num_samples=self.fisher_samples,
        )
        
        # Store teacher model
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Store current optimal params
        self.ewc_loss.register_optimal_params(model)
    
    def consolidate(
        self,
        model: nn.Module,
        plastic_memory,
        dataloader: Iterator,
        task_loss_fn: callable,
    ) -> Dict[str, float]:
        """
        Execute consolidation phase.
        
        Args:
            model: The full model to consolidate
            plastic_memory: PlasticMemory wrapper for LoRA adapters
            dataloader: Training data for consolidation
            task_loss_fn: Task-specific loss function
            
        Returns:
            Dictionary of consolidation metrics
        """
        # Temporarily unfreeze Stable Core
        self._unfreeze_stable_core(model)
        
        # Create optimizer for Stable Core only (not LoRA)
        stable_params = [
            p for n, p in model.named_parameters()
            if 'lora_' not in n.lower() and p.requires_grad
        ]
        optimizer = torch.optim.Adam(stable_params, lr=self.consolidation_lr)
        
        total_loss = 0.0
        ewc_loss_sum = 0.0
        kd_loss_sum = 0.0
        task_loss_sum = 0.0
        
        step = 0
        for batch in dataloader:
            if step >= self.consolidation_steps:
                break
            
            optimizer.zero_grad()
            
            # Move batch to device
            device = next(model.parameters()).device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch) if isinstance(batch, dict) else model(batch)
            
            # Task loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                L_task = outputs.loss
            else:
                L_task = task_loss_fn(outputs, batch)
            
            # EWC loss
            L_ewc = self.ewc_loss(model)
            
            # KD loss (if teacher available)
            L_kd = torch.tensor(0.0, device=device)
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch) if isinstance(batch, dict) else self.teacher_model(batch)
                
                student_logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
                
                L_kd = self.kd_loss(student_logits, teacher_logits)
            
            # Combined loss
            loss = L_task + L_ewc + L_kd
            
            # Backward and update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stable_params, 1.0)
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            ewc_loss_sum += L_ewc.item()
            kd_loss_sum += L_kd.item()
            task_loss_sum += L_task.item()
            
            step += 1
        
        # Re-freeze Stable Core
        self._freeze_stable_core(model)
        
        # Update optimal params for next consolidation
        self.ewc_loss.register_optimal_params(model)
        
        # Partially reset Plastic Memory
        plastic_memory.partial_reset(self.plastic_reset_ratio)
        
        # Cleanup
        self.teacher_model = None
        self.consolidation_count += 1
        
        metrics = {
            'total_loss': total_loss / step if step > 0 else 0,
            'ewc_loss': ewc_loss_sum / step if step > 0 else 0,
            'kd_loss': kd_loss_sum / step if step > 0 else 0,
            'task_loss': task_loss_sum / step if step > 0 else 0,
            'steps': step,
            'consolidation_count': self.consolidation_count,
        }
        
        self.consolidation_losses.append(metrics)
        
        return metrics
    
    def _unfreeze_stable_core(self, model: nn.Module):
        """Temporarily unfreeze Stable Core for consolidation."""
        for name, param in model.named_parameters():
            if 'lora_' not in name.lower():
                param.requires_grad = True
    
    def _freeze_stable_core(self, model: nn.Module):
        """Re-freeze Stable Core after consolidation."""
        for name, param in model.named_parameters():
            if 'lora_' not in name.lower():
                param.requires_grad = False
    
    def validate_forgetting(
        self,
        model: nn.Module,
        validation_dataloaders: Dict[str, Iterator],
        metric_fn: callable,
    ) -> Dict[str, float]:
        """
        Validate model performance on all previous tasks.
        
        Used to check for catastrophic forgetting.
        
        Args:
            model: Model to evaluate
            validation_dataloaders: Dict mapping task names to dataloaders
            metric_fn: Function to compute performance metric
            
        Returns:
            Dict mapping task names to performance metrics
        """
        model.eval()
        results = {}
        
        with torch.no_grad():
            for task_name, dataloader in validation_dataloaders.items():
                metrics = []
                for batch in dataloader:
                    device = next(model.parameters()).device
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()}
                    
                    outputs = model(**batch) if isinstance(batch, dict) else model(batch)
                    metric = metric_fn(outputs, batch)
                    metrics.append(metric)
                
                results[task_name] = sum(metrics) / len(metrics) if metrics else 0.0
        
        model.train()
        return results

