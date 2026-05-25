"""OneTrans trainer with dual optimizer (Adagrad + RMSProp per paper)."""

import os
import glob
import shutil
import logging
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping


class OneTransTrainer:
    """OneTrans trainer for binary classification.

    Paper defaults (Section 4.1.4):
      - Sparse: Adagrad
      - Dense: RMSProp (lr=0.005, alpha=0.99999, momentum=0)
      - No weight decay
      - Gradient clipping: dense=90, sparse=120
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float = 0.005,
        num_epochs: int = 10,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
        early_stopping: Optional[EarlyStopping] = None,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.01,
        grad_clip_dense: float = 90.0,
        grad_clip_sparse: float = 120.0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.schema_path = schema_path

        # Dual optimizer per paper
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sum(p.numel() for p in sparse_params):,} (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {sum(p.numel() for p in dense_params):,} (RMSProp lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(sparse_params, lr=sparse_lr)
            self.dense_optimizer = torch.optim.RMSprop(dense_params, lr=lr, alpha=0.99999, momentum=0)
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99999, momentum=0)

        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        self.early_stopping = early_stopping or EarlyStopping(
            checkpoint_path=os.path.join(save_dir, 'best_model', 'model.pt'),
            patience=5, label='model')
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.grad_clip_dense = grad_clip_dense
        self.grad_clip_sparse = grad_clip_sparse
        self.ckpt_params = ckpt_params or {}
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _save_checkpoint(self, global_step: int, is_best: bool = False,
                         skip_model_file: bool = False) -> str:
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)
        if self.train_config:
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(self.train_config, f, indent=2)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _handle_validation_result(self, total_step: int, val_auc: float, val_logloss: float) -> None:
        old_best = self.early_stopping.best_score
        is_likely_new_best = old_best is None or val_auc > old_best + self.early_stopping.delta
        if not is_likely_new_best:
            self.early_stopping(val_auc, self.model, {"best_val_AUC": val_auc, "best_val_logloss": val_logloss})
            return

        best_dir = os.path.join(self.save_dir, self._build_step_dir_name(total_step, is_best=True))
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        self._remove_old_best_dirs()
        self.early_stopping(val_auc, self.model, {"best_val_AUC": val_auc, "best_val_logloss": val_logloss})

        if self.early_stopping.best_score != old_best and os.path.exists(self.early_stopping.checkpoint_path):
            self._save_checkpoint(total_step, is_best=True, skip_model_file=True)

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _train_step(self, batch: Dict[str, Any]) -> float:
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        seq_data = {}
        seq_lens = {}
        seq_domains = device_batch.get('_seq_domains', [])
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']

        logits = self.model(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data if seq_data else None,
            seq_lens=seq_lens if seq_lens else None,
        ).squeeze(-1)

        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()

        # Gradient clipping per paper (Section 4.1.4)
        if self.sparse_optimizer is not None:
            sparse_params = self.model.get_sparse_params()
            if sparse_params:
                torch.nn.utils.clip_grad_norm_(sparse_params, max_norm=self.grad_clip_sparse, foreach=False)
            dense_params = self.model.get_dense_params()
            if dense_params:
                torch.nn.utils.clip_grad_norm_(dense_params, max_norm=self.grad_clip_dense, foreach=False)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_dense, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return loss.item()

    def _evaluate_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        seq_data = {}
        seq_lens = {}
        seq_domains = device_batch.get('_seq_domains', [])
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']

        logits, _ = self.model.predict(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data if seq_data else None,
            seq_lens=seq_lens if seq_lens else None,
        )
        return logits.squeeze(-1), label

    def train(self) -> None:
        logging.info("Start training (OneTrans)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)
                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()
                    logging.info(f"Step {total_step} Validation | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")
                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                    self._handle_validation_result(total_step, val_auc, val_logloss)
                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader):.4f}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()
            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)
            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        logging.info("Start Evaluation (OneTrans)")
        self.model.eval()

        all_logits, all_labels = [], []
        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if not all_logits:
            return 0.0, float('inf')

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).long()
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            logging.warning(f"[Evaluate] {int(nan_mask.sum())}/{len(probs)} NaN predictions filtered")
            valid = ~nan_mask
            probs, labels_np = probs[valid], labels_np[valid]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item() if len(valid_logits) > 0 else float('inf')

        return auc, logloss