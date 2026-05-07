"""Training utilities for InterFormer."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional


class CTRTrainer:
    """
    Simple training loop for InterFormer.

    Args:
        model     : InterFormer instance
        lr        : learning rate
        weight_decay : L2 regularization
        device    : "cpu" or "cuda"
    """
    def __init__(self, model, lr: float = 1e-3,
                 weight_decay: float = 1e-5, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            dense, sparse_ids, seq_ids, labels = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            logits = self.model(dense, sparse_ids, seq_ids)
            loss = self.criterion(logits, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        for batch in loader:
            dense, sparse_ids, seq_ids, labels = [b.to(self.device) for b in batch]
            logits = self.model(dense, sparse_ids, seq_ids)
            loss = self.criterion(logits, labels.float())
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Compute AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_np, probs)
        except ImportError:
            auc = float("nan")

        return {
            "loss": total_loss / len(loader),
            "auc": auc,
        }

    def fit(self, train_loader, val_loader=None, epochs: int = 10):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            msg = f"Epoch {epoch:3d} | train_loss={train_loss:.4f}"
            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                self.history["val_loss"].append(metrics["loss"])
                self.history["val_auc"].append(metrics["auc"])
                msg += f" | val_loss={metrics['loss']:.4f} | val_auc={metrics['auc']:.4f}"
            print(msg)
        return self.history