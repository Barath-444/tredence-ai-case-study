"""
model.py
--------
Contains:
  - PrunableLinear: A custom linear layer with learnable gate_scores
    that enable dynamic weight pruning during training.
  - SelfPruningNet: A feedforward neural network for CIFAR-10
    classification built entirely from PrunableLinear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate scores.

    Each weight w_ij has a corresponding gate score g_ij.
    During the forward pass:
        gates         = sigmoid(gate_scores)       -> values in (0, 1)
        pruned_weights = weight * gates            -> element-wise mask
        output        = input @ pruned_weights.T + bias

    When a gate value approaches 0, its corresponding weight is
    effectively "pruned" (zeroed out) from the network.

    Gradients flow through both `weight` and `gate_scores` because
    all operations (sigmoid, element-wise mul, matmul) are differentiable.

    Args:
        in_features  (int): Number of input features.
        out_features (int): Number of output features.
        tau          (float): Temperature for sigmoid gating (default 0.5).
    """

    def __init__(self, in_features: int, out_features: int, tau: float = 1.0):
        super(PrunableLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.tau          = tau

        # --- Standard learnable parameters ---
        # Weight matrix: shape (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        # Bias vector: shape (out_features,)
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )
        #edit by barathraj
        # --- Pruning gate scores ---
        # Same shape as weight; will be passed through sigmoid to get gates in (0,1).
        # Initialized to 0.0 so sigmoid(0) = 0.5 — gates start half-open (neutral).
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        # Initialise weights with Kaiming uniform (standard for ReLU nets)
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weight pruning.

        Steps:
          1. Compute gates via sigmoid → values in (0, 1).
          2. Element-wise multiply weights by gates → pruned_weights.
          3. Standard linear transform: x @ pruned_weights.T + bias.
        """
        # Step 1: Convert raw gate scores to gate values in (0, 1) with temperature
        gates = torch.sigmoid(self.gate_scores / self.tau)  # shape: (out, in)

        # Step 2: Apply gates to weights — dead gates zero out weights
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Step 3: Linear transform (identical to F.linear internals)
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from graph) for analysis."""
        return torch.sigmoid(self.gate_scores / self.tau).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """
        Return the fraction of weights whose gate value is below `threshold`.
        A gate < threshold is considered 'pruned'.
        """
        gates = self.get_gates()
        pruned = (gates < threshold).float().sum()
        total  = gates.numel()
        return (pruned / total).item()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ---------------------------------------------------------------------------
# Network Architecture
# ---------------------------------------------------------------------------

class SelfPruningNet(nn.Module):
    """
    Feedforward neural network for CIFAR-10 image classification.

    Architecture:
        Input  : 32 x 32 x 3 = 3072 flattened pixels
        Layer 1: PrunableLinear(3072 → 1024) + BatchNorm + ReLU + Dropout
        Layer 2: PrunableLinear(1024 → 512)  + BatchNorm + ReLU + Dropout
        Layer 3: PrunableLinear(512  → 256)  + BatchNorm + ReLU + Dropout
        Layer 4: PrunableLinear(256  → 128)  + BatchNorm + ReLU
        Output : PrunableLinear(128  → 10)   (logits, no activation)

    All linear layers are PrunableLinear so the entire network can
    self-prune during training.

    Args:
        dropout_rate (float): Dropout probability (default 0.3).
        tau          (float): Temperature for sigmoid gating (default 1.0).
    """

    def __init__(self, dropout_rate: float = 0.3, tau: float = 1.0):
        super(SelfPruningNet, self).__init__()

        # CIFAR-10: 32x32 RGB images → 3072 input features, 10 output classes
        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3072, 1024, tau=tau)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512, tau=tau)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256, tau=tau)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 128, tau=tau)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc_out = PrunableLinear(128, 10, tau=tau)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)                           # (B, 3072)

        x = self.relu(self.bn1(self.fc1(x)))          # (B, 1024)
        x = self.dropout(x)

        x = self.relu(self.bn2(self.fc2(x)))          # (B, 512)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.fc3(x)))          # (B, 256)
        x = self.dropout(x)

        x = self.relu(self.bn4(self.fc4(x)))          # (B, 128)

        x = self.fc_out(x)                            # (B, 10) — raw logits
        return x

    def prunable_layers(self) -> list:
        """Return a list of all PrunableLinear layers in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        Compute the L1 sparsity regularisation loss.

        SparsityLoss = sum of all gate values across every PrunableLinear layer.

        The L1 norm of gates (which are all positive due to sigmoid) drives
        individual gate values toward exactly 0 — producing true sparsity.
        This is in contrast to L2, which shrinks values but rarely reaches 0.
        """
        total_sum   = torch.tensor(0.0, device=next(self.parameters()).device)
        total_count = 0
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores / layer.tau)   # (out, in) ∈ (0,1)
            total_sum   = total_sum + gates.sum()
            total_count = total_count + gates.numel()

        return total_sum / total_count if total_count > 0 else total_sum

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Compute overall sparsity across the entire network.

        Returns:
            float: Fraction of weights whose gate < threshold.
        """
        pruned_count = 0
        total_count  = 0
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            pruned_count += (gates < threshold).sum().item()
            total_count  += gates.numel()
        return pruned_count / total_count if total_count > 0 else 0.0

    def all_gate_values(self) -> torch.Tensor:
        """Collect all gate values from every prunable layer into a single tensor."""
        all_gates = []
        for layer in self.prunable_layers():
            all_gates.append(layer.get_gates().flatten())
        return torch.cat(all_gates)

    def count_parameters(self) -> dict:
        """Return a dict with total and trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
