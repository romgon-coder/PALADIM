"""
PALADIM Continual Learning Benchmark
====================================
Task 5: Sequential NLP task benchmark for testing stability and plasticity.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from .paladim import PALADIM
from .config import PALADIMConfig


@dataclass
class TaskResult:
    """Results for a single task evaluation."""
    task_id: str
    accuracy: float
    f1_score: float
    loss: float


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    task_results: Dict[str, List[TaskResult]]  # task_id -> results after each training phase
    forward_transfer: float  # Average improvement on new tasks
    backward_transfer: float  # Average performance change on old tasks (negative = forgetting)
    average_accuracy: float  # Final average accuracy across all tasks
    forgetting_rate: float  # How much performance dropped on old tasks


class ContinualLearningBenchmark:
    """
    Benchmark suite for evaluating PALADIM on continual learning.
    
    Implements the standard continual learning evaluation protocol:
    1. Train sequentially on tasks T1, T2, ..., Tn
    2. After each task, evaluate on ALL tasks
    3. Compute forgetting, forward/backward transfer metrics
    
    Example:
        >>> benchmark = ContinualLearningBenchmark()
        >>> benchmark.add_task("sentiment", train_data, test_data)
        >>> benchmark.add_task("ner", train_data2, test_data2)
        >>> results = benchmark.run(model)
    """
    
    def __init__(
        self,
        tokenizer=None,
        max_length: int = 128,
        batch_size: int = 16,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.tasks: Dict[str, Dict] = {}
        self.task_order: List[str] = []
        
    def add_task(
        self,
        task_id: str,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_labels: int = 2,
    ):
        """
        Add a task to the benchmark.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset
            test_dataset: Test dataset
            num_labels: Number of output classes
        """
        self.tasks[task_id] = {
            'train': train_dataset,
            'test': test_dataset,
            'num_labels': num_labels,
        }
        self.task_order.append(task_id)
    
    def run(
        self,
        model: PALADIM,
        epochs_per_task: int = 3,
        consolidate_after_each_task: bool = True,
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Run the complete benchmark.
        
        Args:
            model: PALADIM model to evaluate
            epochs_per_task: Training epochs for each task
            consolidate_after_each_task: Whether to consolidate after each task
            verbose: Print progress
            
        Returns:
            BenchmarkResult with all metrics
        """
        all_results: Dict[str, List[TaskResult]] = {tid: [] for tid in self.task_order}
        
        # Performance matrix: R[i][j] = performance on task j after training on task i
        performance_matrix = np.zeros((len(self.task_order), len(self.task_order)))
        
        for train_idx, task_id in enumerate(self.task_order):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training on Task {train_idx + 1}/{len(self.task_order)}: {task_id}")
                print(f"{'='*50}")
            
            # Get task data
            task_data = self.tasks[task_id]
            train_loader = DataLoader(
                task_data['train'],
                batch_size=self.batch_size,
                shuffle=True,
            )
            
            # Train on current task
            train_metrics = model.learn_task(
                dataloader=train_loader,
                num_epochs=epochs_per_task,
                task_id=task_id,
            )
            
            if verbose:
                print(f"Training loss: {train_metrics['avg_loss']:.4f}")
            
            # Consolidation
            if consolidate_after_each_task:
                if verbose:
                    print("Running consolidation...")
                
                # Create validation dataloaders for all seen tasks
                val_loaders = {}
                for tid in self.task_order[:train_idx + 1]:
                    val_loaders[tid] = DataLoader(
                        self.tasks[tid]['test'],
                        batch_size=self.batch_size,
                    )
                
                consolidation_metrics = model.consolidate_knowledge(
                    dataloader=train_loader,
                    validation_dataloaders=val_loaders,
                )
                
                if verbose:
                    print(f"Consolidation loss: {consolidation_metrics['total_loss']:.4f}")
            
            # Evaluate on ALL tasks
            if verbose:
                print("\nEvaluating on all tasks...")
            
            for eval_idx, eval_task_id in enumerate(self.task_order):
                if eval_idx > train_idx:
                    # Haven't seen this task yet
                    performance_matrix[train_idx, eval_idx] = 0
                    continue
                
                test_loader = DataLoader(
                    self.tasks[eval_task_id]['test'],
                    batch_size=self.batch_size,
                )
                
                result = self._evaluate_task(model, eval_task_id, test_loader)
                all_results[eval_task_id].append(result)
                performance_matrix[train_idx, eval_idx] = result.accuracy
                
                if verbose:
                    print(f"  {eval_task_id}: Acc={result.accuracy:.4f}, F1={result.f1_score:.4f}")
        
        # Compute continual learning metrics
        metrics = self._compute_cl_metrics(performance_matrix)
        
        return BenchmarkResult(
            task_results=all_results,
            forward_transfer=metrics['forward_transfer'],
            backward_transfer=metrics['backward_transfer'],
            average_accuracy=metrics['average_accuracy'],
            forgetting_rate=metrics['forgetting_rate'],
        )
    
    def _evaluate_task(
        self,
        model: PALADIM,
        task_id: str,
        dataloader: DataLoader,
    ) -> TaskResult:
        """Evaluate model on a single task."""
        model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Prepare batch
                batch = {
                    k: v.to(next(model.parameters()).device) 
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                outputs = model(**batch)
                
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                num_batches += 1
        
        model.train()
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return TaskResult(
            task_id=task_id,
            accuracy=accuracy,
            f1_score=f1,
            loss=avg_loss,
        )
    
    def _compute_cl_metrics(
        self,
        performance_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute continual learning metrics from performance matrix.
        
        R[i][j] = performance on task j after training on task i
        
        Metrics:
        - Forward Transfer: How much does prior learning help new tasks?
        - Backward Transfer: How much does new learning affect old tasks?
        - Forgetting: Maximum drop in performance for each task
        """
        n_tasks = performance_matrix.shape[0]
        
        # Average final accuracy
        final_row = performance_matrix[-1, :]
        average_accuracy = np.mean(final_row[final_row > 0])
        
        # Backward Transfer (BWT)
        # BWT = (1/(T-1)) * Σ(R[T,i] - R[i,i])
        bwt_sum = 0.0
        for i in range(n_tasks - 1):
            bwt_sum += performance_matrix[-1, i] - performance_matrix[i, i]
        backward_transfer = bwt_sum / (n_tasks - 1) if n_tasks > 1 else 0
        
        # Forgetting (FM)
        # FM = (1/(T-1)) * Σ max_j<T (R[j,i] - R[T,i])
        forgetting = 0.0
        for i in range(n_tasks - 1):
            max_perf = np.max(performance_matrix[:, i])
            forgetting += max(0, max_perf - performance_matrix[-1, i])
        forgetting_rate = forgetting / (n_tasks - 1) if n_tasks > 1 else 0
        
        # Forward Transfer
        # Simplified: average initial performance on new tasks
        # (Before any training on that task)
        forward_transfer = 0.0  # Would need baseline for this
        
        return {
            'average_accuracy': average_accuracy,
            'backward_transfer': backward_transfer,
            'forgetting_rate': forgetting_rate,
            'forward_transfer': forward_transfer,
        }


def create_sample_benchmark_tasks():
    """
    Create sample tasks for testing the benchmark.
    
    Returns datasets for simple synthetic classification tasks.
    """
    from torch.utils.data import TensorDataset
    
    tasks = {}
    
    # Create synthetic tasks with different "domains"
    for i, task_name in enumerate(['task_a', 'task_b', 'task_c']):
        # Random embeddings (simulating pre-tokenized data)
        n_train, n_test = 500, 100
        seq_len = 32
        vocab_size = 1000
        
        # Different random seeds for different distributions
        np.random.seed(42 + i * 100)
        
        train_ids = torch.randint(0, vocab_size, (n_train, seq_len))
        train_labels = torch.randint(0, 2, (n_train,))
        
        test_ids = torch.randint(0, vocab_size, (n_test, seq_len))
        test_labels = torch.randint(0, 2, (n_test,))
        
        # Create dict-style datasets
        class DictDataset(Dataset):
            def __init__(self, input_ids, labels):
                self.input_ids = input_ids
                self.labels = labels
                self.attention_mask = torch.ones_like(input_ids)
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx],
                }
        
        tasks[task_name] = {
            'train': DictDataset(train_ids, train_labels),
            'test': DictDataset(test_ids, test_labels),
        }
    
    return tasks

