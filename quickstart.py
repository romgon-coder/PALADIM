"""
PALADIM Quick Start Script
==========================
Run this to verify your setup works.

Cost: $0 (uses free Hugging Face models and datasets)

Usage:
    python quickstart.py
    
    # Or with options:
    python quickstart.py --model roberta-base --samples 1000
"""

import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="PALADIM Quick Start")
    parser.add_argument("--model", default="distilbert-base-uncased",
                        help="Model name from Hugging Face (free)")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    print("=" * 60)
    print("PALADIM Quick Start")
    print("Pre Adaptive Learning Architecture of Dual-Process Hebbian-MoE")
    print("=" * 60)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model (FREE)
    print(f"\n✓ Loading model: {args.model}")
    print("  (Downloading from Hugging Face - FREE)")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, 
        num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Detect target modules based on architecture
    target_modules = None
    if "distilbert" in args.model.lower():
        target_modules = ["q_lin", "v_lin"]
    elif "roberta" in args.model.lower() or "bert" in args.model.lower():
        target_modules = ["query", "value"]
    elif "gpt" in args.model.lower():
        target_modules = ["c_attn"]
    else:
        target_modules = ["query", "value"]  # Default
    
    # Apply LoRA (Plastic Memory)
    print(f"\n✓ Applying LoRA adapters (Plastic Memory)")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  Frozen: {total - trainable:,} (Stable Core)")

    # Load dataset (FREE)
    print(f"\n✓ Loading IMDB dataset (FREE)")
    try:
        from datasets import load_dataset
        dataset = load_dataset("imdb", split=f"train[:{args.samples}]")
    except ImportError:
        print("  Installing datasets library...")
        import subprocess
        subprocess.run(["pip", "install", "datasets", "-q"])
        from datasets import load_dataset
        dataset = load_dataset("imdb", split=f"train[:{args.samples}]")
    
    print(f"  Samples: {len(dataset)}")

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
    
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Training
    print(f"\n✓ Training (Rapid Learning Phase)")
    model.to(device)
    train_loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-4)

    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    print(f"  Average loss: {avg_loss:.4f}")

    # Test
    print(f"\n✓ Testing")
    model.eval()
    
    test_texts = [
        "This movie was absolutely fantastic! Best film I've seen all year.",
        "Terrible movie. Boring plot, bad acting. Complete waste of time.",
        "It was okay. Nothing special but watchable I guess.",
    ]
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = logits.argmax().item()
            conf = torch.softmax(logits, dim=-1).max().item()
        
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"  {sentiment} ({conf:.0%}): {text[:50]}...")

    print("\n" + "=" * 60)
    print("✅ SUCCESS! PALADIM basics are working.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run full PALADIM: python train.py --run_benchmark")
    print("  2. Add MoE layers for sparse capacity scaling")
    print("  3. Add EWC consolidation for forgetting prevention")
    print("\nTotal cost: $0")


if __name__ == "__main__":
    main()

