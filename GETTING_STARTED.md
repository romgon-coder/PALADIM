# PALADIM: Getting Started Guide

## Cost Summary

| What | Cost | Where |
|------|------|-------|
| Code | **FREE** | This repository |
| Base Models | **FREE** | Hugging Face Hub |
| Datasets | **FREE** | Hugging Face Datasets |
| Local GPU | **FREE** | Your own hardware |
| Google Colab | **FREE** | Limited GPU hours |
| Colab Pro | $10-50/mo | More GPU, A100 access |

**Bottom line: You can run everything for $0 using free resources.**

---

## Option 1: Google Colab (Easiest, Free)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Change runtime to GPU: `Runtime → Change runtime type → T4 GPU`
4. Run the quick start code below

## Option 2: Local Machine

### Requirements
- Python 3.8+
- GPU with 8GB+ VRAM (for small models) or 24GB+ (for 7B models)
- Or CPU-only (slower but works for small models)

### Installation

```bash
# Create virtual environment
python -m venv paladim_env
source paladim_env/bin/activate  # Linux/Mac
# or: paladim_env\Scripts\activate  # Windows

# Install dependencies
pip install torch transformers peft accelerate scikit-learn tqdm datasets
```

---

## Quick Start Code (Copy-Paste Ready)

```python
"""
PALADIM Quick Start - Runs in Colab or locally
Cost: $0 (uses free models and datasets)
"""

# Install (run once)
# !pip install torch transformers peft datasets tqdm -q

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# ============================================================
# 1. CHECK GPU (optional but faster)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 2. LOAD FREE MODEL FROM HUGGING FACE
# ============================================================
# Choose based on your hardware:
# - CPU or small GPU: "distilbert-base-uncased" (66M params)
# - 8GB GPU: "roberta-base" (125M params)
# - 24GB GPU: "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B params)

model_name = "distilbert-base-uncased"  # FREE, small, fast

print(f"Loading {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✓ Model loaded (FREE from Hugging Face)")

# ============================================================
# 3. ADD LORA ADAPTERS (PLASTIC MEMORY)
# ============================================================
# This is the key PALADIM component - LoRA for fast adaptation

lora_config = LoraConfig(
    r=8,                              # Rank (8-64 typical)
    lora_alpha=16,                    # Scaling
    target_modules=["q_lin", "v_lin"], # DistilBERT attention
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

model = get_peft_model(model, lora_config)

# Check: only LoRA params are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✓ LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

# ============================================================
# 4. LOAD FREE DATASET
# ============================================================
print("Loading IMDB dataset...")
dataset = load_dataset("imdb", split="train[:500]")  # Just 500 samples for testing
print(f"✓ Dataset loaded: {len(dataset)} samples (FREE)")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ============================================================
# 5. TRAIN (RAPID LEARNING PHASE)
# ============================================================
model.to(device)
train_loader = DataLoader(tokenized, batch_size=8, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-4)

print("\nTraining (1 epoch)...")
model.train()
total_loss = 0

for batch in tqdm(train_loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    
    outputs = model(**batch)
    loss = outputs.loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()

print(f"✓ Training complete! Avg loss: {total_loss/len(train_loader):.4f}")

# ============================================================
# 6. TEST
# ============================================================
model.eval()
test_texts = [
    "This movie was absolutely fantastic!",
    "Terrible film. Complete waste of time.",
]

print("\nTesting:")
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        pred = model(**inputs).logits.argmax().item()
    
    print(f"  {'👍 Positive' if pred == 1 else '👎 Negative'}: {text}")

print("\n✅ PALADIM basic setup working!")
print("Next: Add MoE layers and EWC consolidation")
```

---

## Free Models to Use

| Model | Size | GPU Needed | Use Case |
|-------|------|------------|----------|
| `distilbert-base-uncased` | 66M | CPU OK | Testing, prototyping |
| `bert-base-uncased` | 110M | 4GB | Classification tasks |
| `roberta-base` | 125M | 6GB | Better NLU |
| `google/flan-t5-small` | 80M | 4GB | Text generation |
| `TinyLlama/TinyLlama-1.1B` | 1.1B | 8GB | Small LLM |
| `microsoft/phi-2` | 2.7B | 12GB | Efficient LLM |
| `mistralai/Mistral-7B-v0.1` | 7B | 24GB | Full LLM |

All are **FREE** to download from Hugging Face!

---

## Free Datasets

```python
from datasets import load_dataset

# Sentiment
imdb = load_dataset("imdb")
sst2 = load_dataset("glue", "sst2")

# NER
conll = load_dataset("conll2003")

# Question Answering
squad = load_dataset("squad")

# Multiple tasks (great for continual learning!)
glue = load_dataset("glue", "mrpc")  # Paraphrase
```

---

## Hardware Requirements

### Minimum (CPU)
- Any modern laptop
- 8GB RAM
- Works with `distilbert-base-uncased`
- Training: ~10 min for 1000 samples

### Recommended (GPU)
- NVIDIA RTX 3060 or better
- 8-12GB VRAM
- Works with models up to 1-3B params
- Training: ~1 min for 1000 samples

### Ideal (for 7B+ models)
- NVIDIA RTX 4090 or A100
- 24-80GB VRAM
- Full PALADIM with large MoE

---

## Google Colab Tips

1. **Free GPU**: `Runtime → Change runtime type → T4 GPU`
2. **Colab disconnects**: Save checkpoints frequently
3. **Memory issues**: Use smaller batch sizes or `gradient_checkpointing=True`
4. **Faster loading**: Mount Google Drive for model caching

```python
# Mount Drive for caching
from google.colab import drive
drive.mount('/content/drive')

# Cache models to Drive
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'
```

---

## Next Steps

1. **Run the quick start** → Verify everything works
2. **Try the full PALADIM** → Use `paladim/train.py`
3. **Add MoE** → Replace FFN with expert layers
4. **Add EWC** → Protect learned weights
5. **Run benchmark** → Test continual learning

Questions? The code is fully documented in the `paladim/` folder.

