"""OneTrans inference script.

Rebuilds the model from ``schema.json`` + ``train_config.json`` at the
checkpoint directory, loads the saved ``model.pt`` weights, and produces
``predictions.json`` under ``EVAL_RESULT_PATH``.

Resolution order for every hyperparameter:
  1. ``train_config.json`` in ``MODEL_OUTPUT_PATH``
  2. Hardcoded fallback (must stay in sync with ``train.py`` defaults)

Environment variables:
    MODEL_OUTPUT_PATH   Checkpoint directory (contains ``model.pt``,
                        ``train_config.json``, and optionally ``schema.json``).
    EVAL_DATA_PATH      Test data directory (``*.parquet`` + ``schema.json``).
    EVAL_RESULT_PATH    Output directory for ``predictions.json``.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PCVRParquetDataset, FeatureSchema
from model import OneTrans

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# ---------------------------------------------------------------------------
# Fallbacks — must stay in sync with one_trans/train.py argparse defaults
# ---------------------------------------------------------------------------
_FALLBACK_MODEL_CFG: Dict[str, Any] = {
    'd_model': 256,
    'emb_dim': 16,
    'num_layers': 6,
    'num_heads': 4,
    'num_ns_tokens': 12,
    'hidden_mult': 4,
    'dropout': 0.0,
    'emb_skip_threshold': 0,
}
_FALLBACK_DATA_CFG: Dict[str, Any] = {
    'batch_size': 32,
    'num_workers': 0,
    'buffer_batches': 5,
    'seq_max_lens': {'domain_a': 50, 'domain_b': 50, 'domain_c': 50, 'domain_d': 50},
}

_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from *model_dir*."""
    path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            cfg = json.load(f)
        logging.info("Loaded train_config from %s", path)
        return cfg
    logging.warning(
        "train_config.json not found in %s — using hardcoded fallbacks", model_dir)
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparams; missing keys fall back to _FALLBACK_MODEL_CFG."""
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        cfg[key] = train_config.get(key, _FALLBACK_MODEL_CFG[key])
    return cfg


def find_ckpt(model_dir: str) -> str:
    """Return the first ``*.pt`` file found recursively in *model_dir*."""
    for root, _dirs, files in os.walk(model_dir):
        for name in files:
            if name.endswith('.pt'):
                return os.path.join(root, name)
    raise FileNotFoundError(
        f"No *.pt file found in {model_dir}. "
        f"Contents: {os.listdir(model_dir)}")


def load_state_dict_strict(model: nn.Module, ckpt_path: str, device: str) -> None:
    """Strict state_dict load; mismatch → fast-fail with clear error."""
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    logging.info("Loaded weights from %s", ckpt_path)


def build_feature_specs(
    schema: FeatureSchema, per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build (vocab_size, offset, length) specs from a FeatureSchema."""
    specs = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def build_model_from_cfg(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    device: str = 'cpu',
    seq_max_lens: Optional[Dict[str, int]] = None,
) -> nn.Module:
    """Rebuild a OneTrans model matching the training architecture."""
    user_int_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(
        "Building OneTrans: d_model=%(d_model)s, emb_dim=%(emb_dim)s, "
        "num_layers=%(num_layers)s, num_heads=%(num_heads)s, "
        "num_ns_tokens=%(num_ns_tokens)s, hidden_mult=%(hidden_mult)s, "
        "dropout=%(dropout)s, emb_skip_threshold=%(emb_skip_threshold)s",
        model_cfg,
    )

    model = OneTrans(
        user_int_feature_specs=user_int_specs,
        item_int_feature_specs=item_int_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        d_model=model_cfg['d_model'],
        emb_dim=model_cfg['emb_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        num_ns_tokens=model_cfg['num_ns_tokens'],
        hidden_mult=model_cfg['hidden_mult'],
        dropout=model_cfg['dropout'],
        emb_skip_threshold=model_cfg['emb_skip_threshold'],
    ).to(device)

    # Eagerly build blocks so state_dict keys exist before weight loading.
    # L_S = sum of per-domain max_lens + (n_domains - 1) SEP tokens
    if dataset.seq_domains:
        if seq_max_lens:
            total_s = sum(seq_max_lens.get(d, 50) for d in dataset.seq_domains)
        else:
            total_s = len(dataset.seq_domains) * 50
        L_S = total_s + max(0, len(dataset.seq_domains) - 1)
    else:
        L_S = 0
    model._build_blocks(L_S, device)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Env vars ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH', '')
    data_dir = os.environ.get('EVAL_DATA_PATH', '')
    result_dir = os.environ.get('EVAL_RESULT_PATH', '')

    if not model_dir:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    if not data_dir:
        raise ValueError("EVAL_DATA_PATH is not set")
    if not result_dir:
        raise ValueError("EVAL_RESULT_PATH is not set")

    os.makedirs(result_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer model_dir (training-time copy), fall back to data_dir ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info("Using schema: %s", schema_path)

    # ---- Resolve hyperparams ----
    train_config = load_train_config(model_dir)
    model_cfg = resolve_model_cfg(train_config)

    batch_size = int(train_config.get('batch_size', _FALLBACK_DATA_CFG['batch_size']))
    num_workers = int(train_config.get('num_workers', _FALLBACK_DATA_CFG['num_workers']))
    buffer_batches = int(train_config.get('buffer_batches', _FALLBACK_DATA_CFG['buffer_batches']))
    seq_max_lens = train_config.get('seq_max_lens', _FALLBACK_DATA_CFG['seq_max_lens'])
    if isinstance(seq_max_lens, str):
        seq_max_lens = {k.strip(): int(v.strip()) for k, v in
                        (pair.split(':') for pair in seq_max_lens.split(','))}

    # ---- Dataset (all row groups as test, no shuffle) ----
    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=None,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )
    logging.info("Test samples: %s", test_dataset.num_rows)

    # ---- Build model ----
    model = build_model_from_cfg(test_dataset, model_cfg, device, seq_max_lens)

    # ---- Load weights ----
    ckpt_path = find_ckpt(model_dir)
    load_state_dict_strict(model, ckpt_path, device)
    model.eval()

    # ---- Inference ----
    all_probs: List[float] = []
    all_user_ids: List[str] = []
    logging.info("Starting inference ...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            user_int = batch['user_int_feats'].to(device, non_blocking=True)
            item_int = batch['item_int_feats'].to(device, non_blocking=True)
            user_dense = batch['user_dense_feats'].to(device, non_blocking=True)
            item_dense = batch['item_dense_feats'].to(device, non_blocking=True)

            seq_data = {}
            seq_lens = {}
            seq_domains = batch.get('_seq_domains', [])
            for domain in seq_domains:
                seq_data[domain] = batch[domain].to(device, non_blocking=True)
                seq_lens[domain] = batch[f'{domain}_len'].to(device, non_blocking=True)

            user_ids = batch['user_id']

            logits, _ = model.predict(
                user_int_feats=user_int,
                item_int_feats=item_int,
                user_dense_feats=user_dense,
                item_dense_feats=item_dense,
                seq_data=seq_data if seq_data else None,
                seq_lens=seq_lens if seq_lens else None,
            )
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info("  Processed %d samples", (batch_idx + 1) * batch_size)

    logging.info("Inference complete: %d predictions", len(all_probs))

    # ---- Save ----
    predictions = {"predictions": dict(zip(all_user_ids, all_probs))}
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info("Saved predictions to %s", output_path)


if __name__ == '__main__':
    main()