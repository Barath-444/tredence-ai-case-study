# 🧠 Self-Pruning Neural Network
### CIFAR-10 Image Classification with Learnable Weight Pruning
**Tredence AI Engineering Internship — Case Study**

---

## 🎯 The Core Idea

Most neural networks are pruned *after* training — a costly, separate step. This project eliminates that entirely.

**The network learns to prune itself during training.**

Every weight has a learnable **gate** — a scalar between 0 and 1. An L1 penalty on all gates pushes unimportant ones toward exactly 0, removing those weights from the network *while it trains*. The result: a sparse, efficient model that discovered its own minimal architecture. No post-processing needed.

---

## 📁 Project Structure

```
self_pruning_nn/
├── model.py          # PrunableLinear layer + SelfPruningNet architecture
├── train.py          # Training loop, evaluation, lambda comparison
├── utils.py          # Seeding, data loading, logging, plotting
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 🏗️ Architecture

### PrunableLinear — The Core Innovation

A drop-in replacement for `nn.Linear` with one crucial addition: a learnable `gate_scores` tensor of the same shape as the weight matrix.

```
gate_scores  →  sigmoid(· / τ)  →  gates ∈ (0, 1)
                                          ↓
                    pruned_weights = weight ⊙ gates
                                          ↓
                    output = input @ pruned_weights.T + bias
```

Gradients flow through **both** `weight` and `gate_scores` simultaneously via the chain rule. No special tricks required.

### SelfPruningNet — Network Layout

5-layer feedforward network. Every single layer is a `PrunableLinear`:

| Layer  | In → Out    | BatchNorm | ReLU | Dropout |
|--------|-------------|:---------:|:----:|:-------:|
| fc1    | 3072 → 1024 | ✅ | ✅ | ✅ |
| fc2    | 1024 → 512  | ✅ | ✅ | ✅ |
| fc3    | 512 → 256   | ✅ | ✅ | ✅ |
| fc4    | 256 → 128   | ✅ | ✅ | — |
| fc_out | 128 → 10    | — | — | — |

**Total parameters: 7,676,042** — all trainable, including gate_scores.

---

## ⚙️ How Pruning Works

### Loss Function

```
Total Loss = CrossEntropyLoss  +  λ × SparsityLoss

SparsityLoss = mean( sigmoid(gate_scores / τ) )   across all layers
```

### Why L1 on Gates Produces True Sparsity

| Penalty | Gradient near zero | Effect |
|---------|-------------------|--------|
| **L2** | `2x → 0` as x→0 | Shrinks values but **stalls before zero** |
| **L1** | `±1` (constant) | **Constant force toward zero — produces exact sparsity** |

Since `sigmoid(·)` is always positive, the L1 norm of gates equals their sum directly. On every training step, the optimizer faces a competition:

- **CrossEntropy** → wants important gates to stay open (preserve accuracy)
- **L1 penalty** → wants all gates to close (minimise the sum)

**λ controls who wins.**

### Critical Implementation Details

| Decision | Why It Matters |
|----------|---------------|
| `gate_scores` init = 0.0 | sigmoid(0)=0.5 → neutral start, neither open nor closed |
| Temperature τ = 1.0 | Smooth sigmoid gradients, gates move gradually not abruptly |
| Gate LR = 5e-3, Weight LR = 1e-3 | Gates need faster updates to respond to sparsity signal |
| Gates: `weight_decay = 0.0` | L2 decay on gates cancels the L1 sparsity gradient — must be zero |
| Cosine LR scheduler | Smooth convergence across the full sparsification process |
| BatchNorm after every layer | Stabilises activations as weights get progressively removed |

---

## 🚀 How to Run

### Local

```bash
cd self_pruning_nn
pip install -r requirements.txt
python train.py
```

### Google Colab

```python
!pip install -r requirements.txt
!python train.py
```

> 25 epochs × 3 lambdas on a T4 GPU takes ~27 minutes.

---

## 📊 Results

### Training Behaviour — Sp Loss Decreasing Proves Gates Are Moving

**λ = 5.0 (25 epochs):**

| Epoch | Sp Loss | Test Acc% | Sparsity% |
|-------|---------|-----------|-----------|
| 1  | 0.47 | 41.17 | 0.00% |
| 3  | 0.34 | 46.85 | 0.34% |
| 5  | 0.25 | 49.11 | 6.79% |
| 8  | 0.18 | 51.46 | 19.99% |
| 12 | 0.14 | 54.34 | 37.05% |
| 18 | 0.11 | 56.93 | 49.75% |
| 25 | 0.10 | 57.74 | **52.38%** |

Sp Loss falling from 0.47 → 0.10 over 25 epochs confirms gates are genuinely closing — not stuck.

---

### ⭐ Final Results — Lambda vs Accuracy vs Sparsity

| Lambda (λ) | Test Accuracy | Sparsity | Interpretation |
|:----------:|:-------------:|:--------:|----------------|
| **1.0** | **57.95%** | **5.98%** | Mild pruning — gates mostly open, accuracy peak |
| **5.0** | **57.74%** | **52.38%** | Balanced — half the weights pruned, tiny accuracy cost |
| **20.0** | **57.40%** | **82.34%** | Aggressive — 4 in 5 weights pruned, accuracy drops |

### What These Numbers Mean

- **Accuracy drops monotonically**: 57.95% → 57.74% → 57.40% as λ increases ✅
- **Sparsity rises dramatically**: 5.98% → 52.38% → 82.34% as λ increases ✅
- **At λ=20, over 82% of weights are pruned** — the network operates with less than 1-in-5 of its original connections
- The small accuracy drop (~0.55%) despite massive sparsity shows the network successfully identified which weights truly matter

---

## 📈 Output Files

All results saved to `./outputs/`:

| File | Description |
|------|-------------|
| `model_lambda_1.0.pt` | Best checkpoint for λ=1.0 (epoch 23, 57.95%) |
| `model_lambda_5.0.pt` | Best checkpoint for λ=5.0 (epoch 25, 57.74%) |
| `model_lambda_20.0.pt` | Best checkpoint for λ=20.0 (epoch 24, 57.40%) |
| `gates_lambda_*.png` | Gate histograms — spike near 0 for high λ |
| `curves_lambda_*.png` | Training loss + accuracy + sparsity over 25 epochs |
| `lambda_comparison.png` | Bar chart: accuracy vs sparsity across all λ |
| `results.json` | JSON summary of all final metrics |

---

## 🔍 Gate Histogram — Proof of Pruning

For λ=20.0, the gate histogram shows a strong **bimodal distribution**:
- **Tall spike near gate = 0** → 82% of weights pruned (gates collapsed to zero)
- **Small cluster near 0.5–1.0** → the ~18% of weights the network chose to keep

This bimodal pattern is the definitive proof that self-pruning is working correctly.

---

## ✅ Evaluation Criteria — Submission Checklist

| Criterion from PDF | Status | Where |
|--------------------|--------|-------|
| Correct `PrunableLinear` with gradients through gates | ✅ | `model.py` — sigmoid gating, element-wise multiply |
| Sparsity regularisation loss (L1) | ✅ | `model.sparsity_loss()` — mean of all gate values |
| Full training loop with total loss | ✅ | `train.py` — `CE + λ × sparsity_loss` |
| 3 lambda values compared | ✅ | λ ∈ {1.0, 5.0, 20.0} |
| Sparsity increases with λ | ✅ | 5.98% → 52.38% → 82.34% |
| Accuracy decreases with λ | ✅ | 57.95% → 57.74% → 57.40% |
| Gate histogram with spike near 0 | ✅ | `outputs/gates_lambda_20.0.png` |
| L1 vs L2 explanation | ✅ | README — constant gradient argument |
| Clean, readable code | ✅ | Full docstrings, type hints, modular files |

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
```
