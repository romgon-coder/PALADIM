"""
PALADIM Training Script
=======================
Example training script for the PALADIM continual learning system.
"""

import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from config import PALADIMConfig
from paladim import PALADIM
from benchmark import ContinualLearningBenchmark, create_sample_benchmark_tasks


def parse_args():
    parser = argparse.ArgumentParser(description="Train PALADIM model")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_benchmark", action="store_true")
    return parser.parse_args()


class SimpleDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=1000, seq_len=64, vocab_size=1000, num_labels=2):
        self.input_ids = torch.randint(0, vocab_size, (size, seq_len))
        self.attention_mask = torch.ones(size, seq_len)
        self.labels = torch.randint(0, num_labels, (size,))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }


def main():
    args = parse_args()
    
    print("="*60)
    print("PALADIM: Pre Adaptive Learning Architecture")
    print("of Dual-Process Hebbian-MoE Schema")
    print("="*60)
    
    # Create config
    config = PALADIMConfig(
        model_name=args.model_name,
        device=args.device,
    )
    config.moe.num_experts = args.num_experts
    config.lora.rank = args.lora_rank
    config.plastic_lr = args.lr
    config.batch_size = args.batch_size
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  MoE Experts: {config.moe.num_experts}")
    print(f"  LoRA Rank: {config.lora.rank}")
    print(f"  Device: {config.device}")
    
    # Initialize model
    print("\nInitializing PALADIM...")
    model = PALADIM(config)
    
    # Print parameter counts
    param_info = model.get_num_parameters()
    print(f"\nParameter counts:")
    print(f"  Total: {param_info['total']:,}")
    print(f"  Trainable (LoRA): {param_info['trainable']:,}")
    print(f"  Trainable ratio: {param_info['trainable_ratio']:.2%}")
    
    if args.run_benchmark:
        print("\n" + "="*60)
        print("Running Continual Learning Benchmark")
        print("="*60)
        
        # Create benchmark
        benchmark = ContinualLearningBenchmark(batch_size=args.batch_size)
        
        # Add synthetic tasks
        tasks = create_sample_benchmark_tasks()
        for task_name, task_data in tasks.items():
            benchmark.add_task(
                task_id=task_name,
                train_dataset=task_data['train'],
                test_dataset=task_data['test'],
            )
        
        # Run benchmark
        results = benchmark.run(
            model=model,
            epochs_per_task=args.epochs,
            consolidate_after_each_task=True,
            verbose=True,
        )
        
        print("\n" + "="*60)
        print("Benchmark Results")
        print("="*60)
        print(f"Average Accuracy: {results.average_accuracy:.4f}")
        print(f"Backward Transfer: {results.backward_transfer:.4f}")
        print(f"Forgetting Rate: {results.forgetting_rate:.4f}")
    
    else:
        # Simple training demo
        print("\n" + "="*60)
        print("Running Simple Training Demo")
        print("="*60)
        
        train_dataset = SimpleDataset(size=500)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        print("\nTraining on Task 1...")
        metrics = model.learn_task(
            dataloader=train_loader,
            num_epochs=args.epochs,
            task_id="task_1",
        )
        print(f"Task 1 - Avg Loss: {metrics['avg_loss']:.4f}")
        
        print("\nRunning consolidation...")
        consolidation_metrics = model.consolidate_knowledge(dataloader=train_loader)
        print(f"Consolidation Loss: {consolidation_metrics['total_loss']:.4f}")
        
        # Second task
        train_dataset2 = SimpleDataset(size=500)
        train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True)
        
        print("\nTraining on Task 2...")
        metrics2 = model.learn_task(
            dataloader=train_loader2,
            num_epochs=args.epochs,
            task_id="task_2",
        )
        print(f"Task 2 - Avg Loss: {metrics2['avg_loss']:.4f}")
        
        print("\nExpert statistics:")
        expert_stats = model.get_expert_statistics()
        print(f"  MoE Layers: {expert_stats['num_moe_layers']}")
        print(f"  Experts per layer: {expert_stats['experts_per_layer']}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

