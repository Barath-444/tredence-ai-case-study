"""
utils.py
--------
Utility functions for:
  - Reproducibility (seed setting)
  - CIFAR-10 data loading with augmentation
  - Sparsity computation helpers
  - Progress logging
  - Result plotting (histogram of gate values, accuracy/sparsity summary)
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for Colab & scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across runs.

    Covers: Python random, NumPy, PyTorch (CPU & CUDA).
    Also enables deterministic cuDNN algorithms at a minor speed cost.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)
    print(f"[Seed] All random seeds set to {seed}")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA if available, else CPU. Prints the selected device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Device] No GPU found — using CPU (training will be slower)")
    return device


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_cifar10_loaders(
    data_dir: str  = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Download and prepare CIFAR-10 train and test DataLoaders.

    Training transform  : RandomCrop + RandomHorizontalFlip + Normalize
    Validation transform: Normalize only (no augmentation)

    CIFAR-10 mean/std are per-channel statistics computed over the training set.

    Args:
        data_dir   : Directory to download/cache data.
        batch_size : Mini-batch size for both loaders.
        num_workers: Parallel workers for data loading.

    Returns:
        (train_loader, test_loader)
    """
    # Per-channel mean and std of CIFAR-10 training set
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),          # random crop with 4px padding
        transforms.RandomHorizontalFlip(),             # random left-right flip
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[Data] CIFAR-10 loaded → "
          f"Train: {len(train_dataset):,} samples | "
          f"Test: {len(test_dataset):,} samples | "
          f"Batch size: {batch_size}")

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Lightweight logger that prints and optionally stores training metrics.

    Usage:
        logger = TrainingLogger(lambda_val=0.01)
        logger.log_epoch(epoch=1, train_loss=1.23, ce_loss=1.10,
                         sp_loss=1300.0, test_acc=45.2, sparsity=0.32)
        logger.print_summary()
    """

    def __init__(self, lambda_val: float):
        self.lambda_val = lambda_val
        self.history: list[dict] = []
        # Print header once
        header = (
            f"\n{'='*80}\n"
            f"  Training with λ = {lambda_val}\n"
            f"{'='*80}\n"
            f"{'Epoch':>6} | {'Train Loss':>10} | {'CE Loss':>9} | "
            f"{'Sp Loss':>10} | {'Test Acc%':>9} | {'Sparsity%':>10}"
        )
        print(header)
        print("-" * 80)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        ce_loss: float,
        sp_loss: float,
        test_acc: float,
        sparsity: float,
    ) -> None:
        record = dict(
            epoch=epoch, train_loss=train_loss, ce_loss=ce_loss,
            sp_loss=sp_loss, test_acc=test_acc, sparsity=sparsity
        )
        self.history.append(record)
        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {ce_loss:>9.4f} | "
            f"{sp_loss:>10.2f} | {test_acc:>9.2f} | {sparsity*100:>9.2f}%"
        )

    def print_summary(self) -> None:
        if not self.history:
            return
        best = max(self.history, key=lambda r: r["test_acc"])
        print(f"\n[Summary λ={self.lambda_val}]  "
              f"Best Test Acc = {best['test_acc']:.2f}%  "
              f"at Epoch {best['epoch']}  |  "
              f"Final Sparsity = {self.history[-1]['sparsity']*100:.2f}%\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gate_histogram(
    gate_values: torch.Tensor,
    lambda_val: float,
    save_path: str,
    test_acc: float,
    sparsity: float,
) -> None:
    """
    Plot and save a histogram of all gate values for a trained model.

    A successful self-pruning model shows:
      - A tall spike near 0  (pruned / dead weights)
      - A cluster of values near 0.5–1.0  (active weights)

    Args:
        gate_values : 1-D tensor of all gate values across the network.
        lambda_val  : Lambda used during training (for title).
        save_path   : File path to save the figure (PNG).
        test_acc    : Final test accuracy (for annotation).
        sparsity    : Final sparsity fraction (for annotation).
    """
    vals = gate_values.cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(vals, bins=100, color="#2563EB", edgecolor="white",
            linewidth=0.3, alpha=0.85)

    # Threshold line
    ax.axvline(x=0.01, color="#DC2626", linestyle="--", linewidth=1.5,
               label="Prune threshold (0.01)")

    ax.set_title(
        f"Gate Value Distribution  |  λ = {lambda_val}\n"
        f"Test Accuracy: {test_acc:.2f}%   |   Sparsity: {sparsity*100:.2f}%",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.set_xlabel("Gate Value  (sigmoid output)", fontsize=11)
    ax.set_ylabel("Number of Weights",            fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved gate histogram → {save_path}")


def plot_lambda_comparison(
    results: list[dict],
    save_path: str,
) -> None:
    """
    Plot a side-by-side bar chart comparing Test Accuracy and Sparsity
    across different lambda values.

    Args:
        results  : List of dicts, each with keys:
                   'lambda', 'test_acc', 'sparsity'
        save_path: File path to save the figure (PNG).
    """
    lambdas   = [str(r["lambda"])    for r in results]
    accs      = [r["test_acc"]       for r in results]
    sparsities= [r["sparsity"] * 100 for r in results]

    x     = np.arange(len(lambdas))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))

    bars1 = ax1.bar(x - width/2, accs,       width, label="Test Accuracy (%)",
                    color="#2563EB", alpha=0.85)
    ax1.set_xlabel("Lambda (λ)",   fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lambdas)
    ax1.set_ylim(0, 100)

    ax2   = ax1.twinx()
    bars2 = ax2.bar(x + width/2, sparsities, width, label="Sparsity (%)",
                    color="#16A34A", alpha=0.85)
    ax2.set_ylabel("Sparsity (%)", fontsize=12, color="#16A34A")
    ax2.tick_params(axis="y", labelcolor="#16A34A")
    ax2.set_ylim(0, 100)

    # Annotate bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=9, color="#1e3a8a")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 fontsize=9, color="#14532d")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("Sparsity vs Accuracy Trade-off Across Lambda Values",
                  fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved lambda comparison → {save_path}")


def plot_training_curves(
    history: list[dict],
    lambda_val: float,
    save_path: str,
) -> None:
    """
    Plot train loss and test accuracy over epochs for a single lambda run.

    Args:
        history   : List of epoch dicts from TrainingLogger.
        lambda_val: Lambda value used (for title).
        save_path : File path for the saved PNG.
    """
    epochs   = [r["epoch"]    for r in history]
    tr_loss  = [r["train_loss"] for r in history]
    test_acc = [r["test_acc"]   for r in history]
    sparsity = [r["sparsity"] * 100 for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, tr_loss, color="#2563EB", linewidth=2)
    ax1.set_title(f"Total Training Loss  (λ={lambda_val})", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, test_acc,  color="#16A34A", linewidth=2, label="Test Acc %")
    ax2_r = ax2.twinx()
    ax2_r.plot(epochs, sparsity, color="#DC2626", linewidth=2,
               linestyle="--", label="Sparsity %")
    ax2.set_title(f"Test Accuracy & Sparsity  (λ={lambda_val})", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy (%)", color="#16A34A")
    ax2_r.set_ylabel("Sparsity (%)",    color="#DC2626")

    lines  = [plt.Line2D([0],[0], color="#16A34A", lw=2),
              plt.Line2D([0],[0], color="#DC2626",  lw=2, linestyle="--")]
    labels = ["Test Acc %", "Sparsity %"]
    ax2.legend(lines, labels, fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved training curves → {save_path}")


# ---------------------------------------------------------------------------
# Results Table
# ---------------------------------------------------------------------------

def print_results_table(results: list[dict]) -> None:
    """
    Print a nicely formatted Markdown-style results table to stdout.

    Args:
        results: List of dicts with keys 'lambda', 'test_acc', 'sparsity'.
    """
    print("\n" + "=" * 55)
    print("  FINAL RESULTS — Lambda vs Accuracy vs Sparsity")
    print("=" * 55)
    print(f"{'Lambda':>10} | {'Test Accuracy':>14} | {'Sparsity':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['lambda']:>10} | {r['test_acc']:>13.2f}% | {r['sparsity']*100:>11.2f}%")
    print("=" * 55 + "\n")
