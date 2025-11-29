# PALADIM

**Pre Adaptive Learning Architecture of Dual-Process Hebbian-MoE Schema**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A continual learning architecture that bridges the gap between transient In-Context Learning and permanent Fine-Tuning. PALADIM scales with compute, not just data.

## Key Features

- **Dual-Process Memory**: Fast plastic adaptation (LoRA) + slow stable consolidation (EWC)
- **Mixture of Experts**: Sparse activation with dynamic expert spawning
- **Catastrophic Forgetting Prevention**: Fisher-weighted protection of important weights
- **Meta-Controller**: Adaptive consolidation timing based on novelty and performance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         PALADIM                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Experience    Plastic      Consolidation     Stable        │
│   Stream   →   Memory   →     Engine      →   Core          │
│              (LoRA)        (EWC+Distill)    (Base)          │
│                                                              │
│      ↓            ↓              ↓             ↓            │
│                                                              │
│  Novelty      Replay         MoE           Meta             │
│  Detector     Buffer        Layer       Controller          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/romgon-coder/PALADIM.git
cd PALADIM
pip install -r requirements.txt
```

### Verify Setup

```bash
python quickstart.py
```

### Basic Usage

```python
from paladim import PALADIM, PALADIMConfig

# Initialize
config = PALADIMConfig(model_name="roberta-base")
model = PALADIM(config)

# Rapid Learning (only LoRA updates)
model.learn_task(train_loader, epochs=3, task_id="sentiment")

# Consolidation (protected transfer to stable core)
model.consolidate_knowledge(train_loader)

# Continue with next task...
model.learn_task(train_loader2, epochs=3, task_id="ner")
```

## Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Stable Core (θ_C)** | `paladim.py` | Pre-trained Transformer base weights (frozen during rapid learning) |
| **Plastic Memory (θ_P)** | `plastic_memory.py` | LoRA adapters for fast Hebbian-like adaptation |
| **MoE Layer** | `moe_layer.py` | Sparse experts with top-k routing and load balancing |
| **Consolidation Engine** | `consolidation.py` | EWC + Knowledge Distillation for protected updates |
| **Meta-Controller** | `meta_controller.py` | Adaptive consolidation triggers |

## Key Equations

**EWC Loss (prevents forgetting):**
```
L_ewc = Σᵢ (λ/2) · Fᵢ · (θᵢ - θ*ᵢ)²
```

**Knowledge Distillation:**
```
L_kd = KL(softmax(z_teacher/T) || softmax(z_student/T)) · T²
```

**MoE Load Balancing:**
```
L_aux = α · N · Σ(fᵢ · Pᵢ)
```

## Continual Learning Benchmark

```bash
python train.py --run_benchmark
```

Metrics reported:
- **Forward Transfer**: How prior learning helps new tasks
- **Backward Transfer**: Impact on previously learned tasks
- **Forgetting Rate**: Maximum performance drop on old tasks
- **Average Accuracy**: Final performance across all tasks

## Configuration

```python
config = PALADIMConfig(
    # Base model
    model_name="roberta-base",
    
    # MoE settings
    moe=MoEConfig(
        num_experts=8,
        top_k=2,
        load_balance_weight=0.01,
    ),
    
    # LoRA settings  
    lora=LoRAConfig(
        rank=16,
        alpha=32,
        target_modules=["query", "value"],
    ),
    
    # Consolidation
    consolidation=ConsolidationConfig(
        ewc_lambda=5000,
        kd_temperature=2.0,
        consolidation_steps=100,
    ),
)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.5+

## Hardware

| Model Size | GPU Memory | Example |
|------------|------------|---------|
| 66M (DistilBERT) | 2GB / CPU | Testing |
| 125M (RoBERTa) | 4-6GB | Development |
| 1-3B | 8-12GB | Small LLM |
| 7B+ | 24GB+ | Full LLM |

## References

- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2017
- [LoRA](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [Mixture of Experts](https://arxiv.org/abs/1701.06538) - Shazeer et al., 2017
- [Complementary Learning Systems](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(16)30043-2) - Kumaran et al., 2016

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{paladim2024,
  title={PALADIM: Pre Adaptive Learning Architecture of Dual-Process Hebbian-MoE Schema},
  author={romgon-coder},
  year={2024},
  url={https://github.com/romgon-coder/PALADIM}
}
```

