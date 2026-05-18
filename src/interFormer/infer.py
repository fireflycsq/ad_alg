"""InterFormer inference script.

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
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import InterFormerParquetDataset, get_interformer_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# ---------------------------------------------------------------------------
# Fallbacks — must stay in sync with train.py argparse defaults
# ---------------------------------------------------------------------------
_FALLBACK_MODEL_CFG: Dict[str, Any] = {
    'embed_dim': 64,
    'n_layers': 3,
    'interaction': 'dhen',
    'n_heads': 8,
    'n_cls_tokens': 4,
    'n_pma_tokens': 2,
    'n_recent_tokens': 2,
    'seq_len': 500,
    'dropout': 0.1,
    'mlp_hidden_dims': '256,128',
}
_FALLBACK_DATA_CFG: Dict[str, Any] = {
    'batch_size': 256,
    'num_workers': 0,
    'seq_vocab_size': 100000,
    'item_id_vocab_size': 100000,
    'max_dense_per_feat': 0,
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
    """Return the first ``*.pt`` file in *model_dir*."""
    for name in os.listdir(model_dir):
        if name.endswith('.pt'):
            return os.path.join(model_dir, name)
    raise FileNotFoundError(
        f"No *.pt file found in {model_dir}. "
        f"Contents: {os.listdir(model_dir)}")


def load_state_dict_strict(model: nn.Module, ckpt_path: str, device: str) -> None:
    """Strict state_dict load; mismatch → fast-fail with clear error."""
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    logging.info("Loaded weights from %s", ckpt_path)


def build_model_from_cfg(
    dataset: InterFormerParquetDataset,
    model_cfg: Dict[str, Any],
    device: str = 'cpu',
) -> nn.Module:
    """Rebuild an InterFormer matching the training architecture."""
    from model import InterFormer  # deferred import — avoids ckpt coupling

    mlp_dims = [int(x.strip()) for x in model_cfg['mlp_hidden_dims'].split(',')]

    logging.info(
        "Building InterFormer: embed_dim=%(embed_dim)s, n_layers=%(n_layers)s, "
        "interaction=%(interaction)s, n_heads=%(n_heads)s, "
        "n_cls=%(n_cls_tokens)s, n_pma=%(n_pma_tokens)s, "
        "n_recent=%(n_recent_tokens)s, seq_len=%(seq_len)s",
        model_cfg,
    )

    model = InterFormer(
        dense_dim=dataset.dense_dim,
        sparse_vocab_sizes=dataset.sparse_vocabs,
        seq_len=model_cfg['seq_len'],
        seq_vocab_sizes=dataset.seq_vocab_sizes,
        embed_dim=model_cfg['embed_dim'],
        n_layers=model_cfg['n_layers'],
        interaction=model_cfg['interaction'],
        n_heads=model_cfg['n_heads'],
        n_cls_tokens=model_cfg['n_cls_tokens'],
        n_pma_tokens=model_cfg['n_pma_tokens'],
        n_recent_tokens=model_cfg['n_recent_tokens'],
        n_sequences=dataset.n_sequences,
        sparse_is_array=dataset.sparse_is_array,
        sparse_multi_dim=dataset.sparse_multi_dim,
        dropout=model_cfg['dropout'],
        mlp_hidden_dims=mlp_dims,
    ).to(device)
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
    seq_vocab_size = int(train_config.get('seq_vocab_size', _FALLBACK_DATA_CFG['seq_vocab_size']))
    item_id_vocab_size = int(train_config.get('item_id_vocab_size', _FALLBACK_DATA_CFG['item_id_vocab_size']))
    max_dense_per_feat = int(train_config.get('max_dense_per_feat', _FALLBACK_DATA_CFG['max_dense_per_feat']))

    # ---- Dataset (single-parquet inference: all rows as test) ----
    test_dataset = InterFormerParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_len=model_cfg['seq_len'],
        seq_vocab_size=seq_vocab_size,
        max_dense_per_feat=max_dense_per_feat,
        item_id_vocab_size=item_id_vocab_size,
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
    model = build_model_from_cfg(test_dataset, model_cfg, device)

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
            dense = batch['dense'].to(device, non_blocking=True)
            sparse_ids = batch['sparse_ids'].to(device, non_blocking=True)
            seq_ids = batch['seq_ids'].to(device, non_blocking=True)
            seq_mask = batch['seq_padding_mask'].to(device, non_blocking=True)
            sparse_multi = batch.get('sparse_multi')
            sparse_multi_mask = batch.get('sparse_multi_mask')
            if sparse_multi is not None:
                sparse_multi = sparse_multi.to(device, non_blocking=True)
                sparse_multi_mask = sparse_multi_mask.to(device, non_blocking=True)

            user_ids = batch['user_id']

            logits = model(dense, sparse_ids, seq_ids, seq_mask,
                           sparse_multi, sparse_multi_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
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