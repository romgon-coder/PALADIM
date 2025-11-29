"""
PALADIM Meta-Controller
=======================
Task 4: Adaptive control of learning dynamics and consolidation timing.

Decides WHEN and HOW to consolidate based on:
- Novelty detection (distribution shift)
- Performance monitoring (loss plateau)
- Expert load distribution
- Training progress
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from collections import deque
import numpy as np


class NoveltyDetector:
    """
    Detects distribution shifts in input data.
    
    Uses centroid-based clustering to identify when new
    data is significantly different from previously seen data.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_centroids: int = 10,
        novelty_threshold: float = 0.8,
        momentum: float = 0.99,
    ):
        self.embedding_dim = embedding_dim
        self.num_centroids = num_centroids
        self.novelty_threshold = novelty_threshold
        self.momentum = momentum
        
        # Initialize centroids
        self.centroids: List[torch.Tensor] = []
        self.centroid_counts: List[int] = []
        
        # Running statistics
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        
    def update(self, embeddings: torch.Tensor):
        """
        Update centroids with new embeddings.
        
        Args:
            embeddings: [batch_size, embedding_dim] tensor
        """
        embeddings = embeddings.detach()
        
        # Update running statistics
        if self.mean is None:
            self.mean = embeddings.mean(dim=0)
            self.std = embeddings.std(dim=0) + 1e-6
        else:
            batch_mean = embeddings.mean(dim=0)
            batch_std = embeddings.std(dim=0) + 1e-6
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.std = self.momentum * self.std + (1 - self.momentum) * batch_std
        
        # Normalize embeddings
        normalized = (embeddings - self.mean) / self.std
        
        # Update or add centroids
        for emb in normalized:
            if len(self.centroids) == 0:
                self.centroids.append(emb.clone())
                self.centroid_counts.append(1)
            else:
                # Find nearest centroid
                distances = torch.stack([
                    torch.norm(emb - c) for c in self.centroids
                ])
                min_dist, min_idx = distances.min(dim=0)
                
                if min_dist > self.novelty_threshold and len(self.centroids) < self.num_centroids:
                    # New centroid
                    self.centroids.append(emb.clone())
                    self.centroid_counts.append(1)
                else:
                    # Update existing centroid
                    count = self.centroid_counts[min_idx]
                    self.centroids[min_idx] = (
                        self.centroids[min_idx] * count + emb
                    ) / (count + 1)
                    self.centroid_counts[min_idx] += 1
    
    def compute_novelty(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty score for embeddings.
        
        High novelty = far from all centroids = potential distribution shift.
        
        Args:
            embeddings: [batch_size, embedding_dim] tensor
            
        Returns:
            Novelty scores [batch_size] in range [0, 1]
        """
        if len(self.centroids) == 0 or self.mean is None:
            return torch.ones(embeddings.shape[0], device=embeddings.device)
        
        embeddings = embeddings.detach()
        normalized = (embeddings - self.mean) / self.std
        
        # Compute minimum distance to any centroid
        min_distances = torch.zeros(embeddings.shape[0], device=embeddings.device)
        for i, emb in enumerate(normalized):
            distances = torch.stack([torch.norm(emb - c) for c in self.centroids])
            min_distances[i] = distances.min()
        
        # Normalize to [0, 1] using sigmoid
        novelty = torch.sigmoid(min_distances - self.novelty_threshold)
        
        return novelty


class PerformanceMonitor:
    """
    Monitors training performance and detects plateaus.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        plateau_threshold: float = 0.005,
        patience: int = 50,
    ):
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.patience = patience
        
        self.loss_history: deque = deque(maxlen=window_size * 2)
        self.best_loss: float = float('inf')
        self.steps_without_improvement: int = 0
        
    def update(self, loss: float) -> Dict[str, float]:
        """
        Update with new loss value.
        
        Returns:
            Dictionary with monitoring statistics
        """
        self.loss_history.append(loss)
        
        # Compute moving average
        if len(self.loss_history) >= self.window_size:
            recent = list(self.loss_history)[-self.window_size:]
            earlier = list(self.loss_history)[-self.window_size*2:-self.window_size]
            
            recent_avg = np.mean(recent)
            earlier_avg = np.mean(earlier) if earlier else recent_avg
            
            improvement = earlier_avg - recent_avg
            
            if improvement > self.plateau_threshold:
                self.steps_without_improvement = 0
                if recent_avg < self.best_loss:
                    self.best_loss = recent_avg
            else:
                self.steps_without_improvement += 1
        else:
            recent_avg = np.mean(list(self.loss_history))
            improvement = 0.0
        
        return {
            'loss_ma': recent_avg,
            'improvement': improvement,
            'steps_without_improvement': self.steps_without_improvement,
            'is_plateau': self.steps_without_improvement >= self.patience,
        }
    
    def is_plateau(self) -> bool:
        """Check if training has plateaued."""
        return self.steps_without_improvement >= self.patience
    
    def reset(self):
        """Reset monitoring state."""
        self.loss_history.clear()
        self.best_loss = float('inf')
        self.steps_without_improvement = 0


class MetaController:
    """
    Meta-Learning Controller for PALADIM.
    
    Orchestrates consolidation decisions based on multiple signals:
    - Training loss patterns
    - Novelty detection
    - Expert load distribution
    - Time-based triggers
    
    Implements adaptive learning rate modulation and consolidation scheduling.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        consolidation_interval: int = 500,
        novelty_threshold: float = 0.8,
        loss_plateau_patience: int = 50,
        loss_plateau_threshold: float = 0.005,
        expert_load_threshold: float = 0.95,
    ):
        self.consolidation_interval = consolidation_interval
        self.novelty_threshold = novelty_threshold
        self.expert_load_threshold = expert_load_threshold
        
        # Sub-components
        self.novelty_detector = NoveltyDetector(
            embedding_dim=embedding_dim,
            novelty_threshold=novelty_threshold,
        )
        self.performance_monitor = PerformanceMonitor(
            patience=loss_plateau_patience,
            plateau_threshold=loss_plateau_threshold,
        )
        
        # State tracking
        self.step_count: int = 0
        self.last_consolidation_step: int = 0
        self.consolidation_count: int = 0
        self.consolidation_cooldown: int = 0
        
        # Decision history
        self.decision_history: List[Dict] = []
        
    def step(
        self,
        loss: float,
        embeddings: Optional[torch.Tensor] = None,
        expert_loads: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Process one training step and make decisions.
        
        Args:
            loss: Current training loss
            embeddings: Optional hidden state embeddings for novelty detection
            expert_loads: Optional MoE expert utilization tensor
            
        Returns:
            Dictionary containing decisions and metrics
        """
        self.step_count += 1
        
        # Update cooldown
        if self.consolidation_cooldown > 0:
            self.consolidation_cooldown -= 1
        
        # Update performance monitor
        perf_stats = self.performance_monitor.update(loss)
        
        # Compute novelty if embeddings provided
        novelty_score = 0.0
        if embeddings is not None:
            novelty = self.novelty_detector.compute_novelty(embeddings)
            novelty_score = novelty.mean().item()
            self.novelty_detector.update(embeddings)
        
        # Check expert loads
        expert_overload = False
        max_expert_load = 0.0
        if expert_loads is not None:
            max_expert_load = expert_loads.max().item()
            expert_overload = max_expert_load > self.expert_load_threshold
        
        # Consolidation decision logic
        should_consolidate = self._should_consolidate(
            perf_stats=perf_stats,
            novelty_score=novelty_score,
            expert_overload=expert_overload,
        )
        
        # Expert spawning decision
        should_spawn = self._should_spawn_expert(
            novelty_score=novelty_score,
            expert_loads=expert_loads,
        )
        
        # Learning rate modulation
        lr_multiplier = self._compute_lr_multiplier(novelty_score)
        
        # Replay ratio adjustment
        replay_ratio = self._compute_replay_ratio(perf_stats)
        
        decisions = {
            'consolidate': should_consolidate,
            'spawn_expert': should_spawn,
            'lr_multiplier': lr_multiplier,
            'replay_ratio': replay_ratio,
            'novelty_score': novelty_score,
            'max_expert_load': max_expert_load,
            'loss_ma': perf_stats['loss_ma'],
            'is_plateau': perf_stats['is_plateau'],
            'step': self.step_count,
        }
        
        self.decision_history.append(decisions)
        
        return decisions
    
    def _should_consolidate(
        self,
        perf_stats: Dict,
        novelty_score: float,
        expert_overload: bool,
    ) -> bool:
        """Determine if consolidation should be triggered."""
        if self.consolidation_cooldown > 0:
            return False
        
        steps_since_consolidation = self.step_count - self.last_consolidation_step
        
        # Trigger conditions (any one sufficient)
        triggers = [
            # Time-based: consolidate after fixed interval
            steps_since_consolidation >= self.consolidation_interval,
            
            # Performance-based: consolidate on plateau
            perf_stats['is_plateau'],
            
            # Novelty-based: consolidate when novelty DROPS (learned the new distribution)
            novelty_score < 0.2 and steps_since_consolidation > 100,
        ]
        
        return any(triggers)
    
    def _should_spawn_expert(
        self,
        novelty_score: float,
        expert_loads: Optional[torch.Tensor],
    ) -> bool:
        """Determine if a new expert should be spawned."""
        # High novelty indicates new domain
        if novelty_score > self.novelty_threshold:
            return True
        
        # Expert overload indicates capacity needed
        if expert_loads is not None:
            if expert_loads.max().item() > self.expert_load_threshold:
                return True
        
        return False
    
    def _compute_lr_multiplier(self, novelty_score: float) -> float:
        """Compute learning rate multiplier based on novelty."""
        if novelty_score > 0.6:
            return 1.5  # High novelty: faster learning
        elif novelty_score < 0.3:
            return 0.7  # Low novelty: careful updates
        return 1.0
    
    def _compute_replay_ratio(self, perf_stats: Dict) -> float:
        """Compute replay ratio for mixing old and new data."""
        # Increase replay when performance is degrading
        if perf_stats['steps_without_improvement'] > 20:
            return 0.4  # More replay
        return 0.25  # Default ratio
    
    def on_consolidation_complete(self):
        """Called after consolidation finishes."""
        self.consolidation_count += 1
        self.last_consolidation_step = self.step_count
        self.consolidation_cooldown = 50  # Prevent immediate re-trigger
        self.performance_monitor.reset()
    
    def get_statistics(self) -> Dict:
        """Get controller statistics."""
        return {
            'total_steps': self.step_count,
            'consolidation_count': self.consolidation_count,
            'steps_since_consolidation': self.step_count - self.last_consolidation_step,
            'cooldown_remaining': self.consolidation_cooldown,
        }

