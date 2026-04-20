"""
train.py
--------
Main training script for the Self-Pruning Neural Network.

What this script does:
  1. Loads CIFAR-10 data with augmentation.
  2. Trains SelfPruningNet for each lambda in LAMBDA_VALUES.
  3. For each run:
       - Computes Total Loss = CrossEntropy + lambda * SparsityLoss
       - Evaluates test accuracy and sparsity every epoch
       - Saves the best model checkpoint
       - Generates gate histogram and training curve plots
  4. Prints a final comparison table.
  5. Generates a lambda comparison bar chart.

Run:
    python train.py

Google Colab:
    Upload the project folder, then:
        !pip install -r requirements.txt
        !python train.py
"""

import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim

from model import SelfPruningNet
from utils import (
    set_seed,
    get_device,
    get_cifar10_loaders,
    TrainingLogger,
    plot_gate_histogram,
    plot_lambda_comparison,
    plot_training_curves,
    print_results_table,
)
#edit by barathraj
# ---------------------------------------------------------------------------
# ─── Hyperparameters ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Lambda values to compare (controls sparsity vs accuracy trade-off)
# Scaled for MEAN-based sparsity loss (Fix from research).
LAMBDA_VALUES = [1.0, 5.0, 20.0]

# Training config
NUM_EPOCHS      = 25          # 25 epochs is enough for this experiment
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-3
GATE_LEARNING_RATE = 5e-3     # Set to 5e-3 
WEIGHT_DECAY    = 1e-4        # L2 regularisation on standard params
DROPOUT_RATE    = 0.3
SEED            = 42
PRUNE_THRESHOLD = 0.05        # Set to 0.05 

# Paths
DATA_DIR    = "./data"
SAVE_DIR    = "./outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# ─── Training for one epoch ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SelfPruningNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    lambda_val: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Run one full training epoch.

    Loss decomposition:
        total_loss = cross_entropy_loss + lambda_val * sparsity_loss

    The sparsity_loss is the L1 norm of all gate values — encouraging
    gates to collapse to zero and prune their associated weights.

    Returns:
        (avg_total_loss, avg_ce_loss, avg_sp_loss) over all batches.
    """
    model.train()
    total_loss_sum = 0.0
    ce_loss_sum    = 0.0
    sp_loss_sum    = 0.0
    n_batches      = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)                         # (B, 10)

        # Classification loss
        ce_loss = criterion(logits, labels)

        # Sparsity regularisation loss (L1 of all gates)
        sp_loss = model.sparsity_loss()

        # Combined loss
        loss = ce_loss + lambda_val * sp_loss

        # Backward pass — gradients flow through weight AND gate_scores
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        ce_loss_sum    += ce_loss.item()
        sp_loss_sum    += sp_loss.item()
        n_batches      += 1

    return (total_loss_sum / n_batches,
            ce_loss_sum    / n_batches,
            sp_loss_sum    / n_batches)


# ---------------------------------------------------------------------------
# ─── Evaluation ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the given loader.

    Returns:
        Accuracy (%) on the dataset.
    """
    model.eval()
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds  = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# ─── Full Training Run for One Lambda ───────────────────────────────────────
# ---------------------------------------------------------------------------

def run_experiment(
    lambda_val: float,
    train_loader,
    test_loader,
    device: torch.device,
) -> dict:
    """
    Train SelfPruningNet with a specific lambda value and return results.

    Saves:
      - Best model checkpoint: outputs/model_lambda_{lambda_val}.pt
      - Gate histogram plot:   outputs/gates_lambda_{lambda_val}.png
      - Training curve plot:   outputs/curves_lambda_{lambda_val}.png

    Args:
        lambda_val   : Sparsity regularisation coefficient.
        train_loader : DataLoader for training set.
        test_loader  : DataLoader for test set.
        device       : torch.device (CPU or CUDA).

    Returns:
        dict with keys: 'lambda', 'test_acc', 'sparsity', 'history'
    """
    # Fresh model for each experiment (no cross-contamination between runs)
    set_seed(SEED)
    model = SelfPruningNet(dropout_rate=DROPOUT_RATE).to(device)

    param_info = model.count_parameters()
    print(f"\n[Model] Parameters → Total: {param_info['total']:,} | "
          f"Trainable: {param_info['trainable']:,}")

    # Optimizer — separate groups for weights/biases and gate_scores
    gate_params  = []
    other_params = []
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.Adam([
        {"params": other_params, "lr": LEARNING_RATE,      "weight_decay": WEIGHT_DECAY},
        {"params": gate_params,  "lr": GATE_LEARNING_RATE, "weight_decay": 0.0}
    ])

    # Cosine annealing: smoothly reduces LR from LEARNING_RATE → 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    criterion = nn.CrossEntropyLoss()
    logger    = TrainingLogger(lambda_val=lambda_val)

    best_acc      = 0.0
    best_ckpt_path = os.path.join(SAVE_DIR, f"model_lambda_{lambda_val}.pt")

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──
        train_loss, ce_loss, sp_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_val, device
        )

        # ── Evaluate ──
        test_acc = evaluate(model, test_loader, device)
        sparsity = model.overall_sparsity(threshold=PRUNE_THRESHOLD)

        # ── Log ──
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            ce_loss=ce_loss,
            sp_loss=sp_loss,
            test_acc=test_acc,
            sparsity=sparsity,
        )

        # ── Save best model ──
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch":      epoch,
                "lambda":     lambda_val,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "test_acc":   test_acc,
                "sparsity":   sparsity,
            }, best_ckpt_path)

        scheduler.step()

    elapsed = time.time() - start_time
    logger.print_summary()
    print(f"[Time] Training took {elapsed:.1f}s  ({elapsed/NUM_EPOCHS:.1f}s/epoch)")
    print(f"[Checkpoint] Best model saved → {best_ckpt_path}")

    # ── Final metrics (from the best checkpoint) ──
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity(threshold=PRUNE_THRESHOLD)
    gate_values    = model.all_gate_values()

    # ── Plots ──
    plot_gate_histogram(
        gate_values=gate_values,
        lambda_val=lambda_val,
        save_path=os.path.join(SAVE_DIR, f"gates_lambda_{lambda_val}.png"),
        test_acc=final_acc,
        sparsity=final_sparsity,
    )
    plot_training_curves(
        history=logger.history,
        lambda_val=lambda_val,
        save_path=os.path.join(SAVE_DIR, f"curves_lambda_{lambda_val}.png"),
    )

    return {
        "lambda":    lambda_val,
        "test_acc":  final_acc,
        "sparsity":  final_sparsity,
        "history":   logger.history,
    }


# ---------------------------------------------------------------------------
# ─── Main ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  Self-Pruning Neural Network — CIFAR-10 Case Study")
    print("  Tredence AI Engineering Internship")
    print("=" * 70)

    # Setup
    set_seed(SEED)
    device = get_device()

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    # Run one experiment per lambda value
    all_results = []
    for lam in LAMBDA_VALUES:
        result = run_experiment(
            lambda_val=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
        all_results.append(result)

    # ── Final comparison table ──
    summary = [{"lambda": r["lambda"],
                "test_acc": r["test_acc"],
                "sparsity": r["sparsity"]}
               for r in all_results]
    print_results_table(summary)

    # ── Lambda comparison bar chart ──
    plot_lambda_comparison(
        results=summary,
        save_path=os.path.join(SAVE_DIR, "lambda_comparison.png"),
    )

    # ── Save results as JSON ──
    json_path = os.path.join(SAVE_DIR, "results.json")
    with open(json_path, "w") as f:
        # history contains floats — serialisable
        json.dump(summary, f, indent=2)
    print(f"[Results] Saved to {json_path}")

    print("\n[Done] All experiments complete. Check the ./outputs/ folder.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
