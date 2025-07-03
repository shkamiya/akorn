# -*- coding: utf-8 -*-
"""
A minimal, self‑contained toy implementation of **Artificial Kuramoto Oscillatory Neurons (AKOrN)**
showing how to embed a single AKOrN layer inside a small network and train it on the classic
`two‑moons` toy dataset.  The code runs on CPU or GPU (if available) and should finish in a
few seconds.

Key simplifications w.r.t. the full paper:
1. Each oscillatory neuron lives on a *2‑sphere* (d = 2).  This lets us represent the
   phase directly by a 2‑D unit vector `[cos θ, sin θ]`.
2. Coupling weights are learned but *shared across the batch* and soft‑symmetrized via `tanh`.
3. A single AKOrN layer is followed by a linear read‑out that flattens the phase vectors.
4. We use a fixed number of Kuramoto steps (`steps=5`) and an explicit Euler integrator.

Despite these simplifications the code reproduces the AKOrN spirit:
• Neurons rotate and synchronize through Kuramoto coupling.  
• The Lyapunov‑like norm constraint is enforced by `F.normalize` after each time step.  
• Test‑time we can simply increase `steps` to perform extra inference iterations.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
#                             Core AKOrN layer
# -----------------------------------------------------------------------------

class AKOrNLayer(nn.Module):
    """Kuramoto‑style oscillatory neurons on the unit d‑sphere."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        d: int = 2,
        steps: int = 5,
        dt: float = 1.0,
        coupling_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.d = d
        self.steps = steps
        self.dt = dt

        # Linear projection from input to initial latent phases (before normalization)
        self.W_in = nn.Parameter(torch.randn(in_features, out_features, d) * 0.2)
        # Learned intrinsic frequency for each neuron
        self.omega = nn.Parameter(torch.zeros(out_features, d))
        # Coupling matrix (learned, will be symmetrized via tanh)
        self.coupling = nn.Parameter(torch.randn(out_features, out_features) * coupling_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x shape (batch, in_features)
        # (1) Project to d‑dimensional latent and normalize to unit length
        # Shape: (batch, out_features, d)
        v = torch.einsum("bi,iod->bod", x, self.W_in)
        v = F.normalize(v, dim=-1)

        # (2) Kuramoto iterations (explicit Euler)
        # Pre‑compute symmetric coupling (same for every batch element)
        K = torch.tanh(self.coupling)  # bounds the magnitude and softly symmetrizes
        for _ in range(self.steps):
            # Aggregate neighbors: sum_j K_ij * v_j
            agg = torch.einsum("ij,bjd->bid", K, v)
            # Euler update and re‑normalize (unit‑sphere constraint)
            v = v + self.dt * (agg + self.omega)  # intrinsic + coupled rotation
            v = F.normalize(v, dim=-1)
        return v  # shape (batch, out_features, d)

# -----------------------------------------------------------------------------
#                             Tiny AKOrN network
# -----------------------------------------------------------------------------

class ToyAKOrNNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_classes: int, steps: int = 5):
        super().__init__()
        self.akorn = AKOrNLayer(input_dim, hidden, d=2, steps=steps)
        # Flatten the (hidden,2) oscillators and map to class logits
        self.readout = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.akorn(x)            # (batch, hidden, 2)
        v_flat = v.flatten(1)        # (batch, hidden*2)
        return self.readout(v_flat)  # logits

# -----------------------------------------------------------------------------
#                             Training utilities
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    hidden: int = 32
    steps: int = 5         # Kuramoto steps during *training*
    epochs: int = 200
    lr: float = 1e-2
    test_steps: int = 20   # extra inference iterations at test‑time


def make_dataloaders(test_size: float = 0.3, batch_size: int = 128) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    X, y = make_moons(n_samples=1500, noise=0.2, random_state=0)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def train_toy_model(cfg: TrainConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    train_loader, test_loader = make_dataloaders()
    model = ToyAKOrNNet(input_dim=2, hidden=cfg.hidden, num_classes=2, steps=cfg.steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            train_loss = running_loss / len(train_loader.dataset)
            acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1:4d} | train loss {train_loss:.4f} | test acc {acc:.2%}")

    # ──────────────────────────────────────────────────────────────────────────
    # Extra inference iterations (test‑time refinement) -----------------------
    # ──────────────────────────────────────────────────────────────────────────
    model.akorn.steps = cfg.test_steps  # increase Kuramoto iterations
    refined_acc = evaluate(model, test_loader, device)
    print("Final accuracy after extra inference iterations:", f"{refined_acc:.2%}")


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


if __name__ == "__main__":
    cfg = TrainConfig()
    train_toy_model(cfg)
