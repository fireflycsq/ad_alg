"""InterFormer inference script (uploaded into the evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``train_config.json``. All model hyperparameters are
resolved first from the ckpt directory's ``train_config.json``, falling back
to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent with the CLI
defaults in ``train.py``).

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (contains ``model.pt`` and
                       ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Output directory for ``predictions.json``.
"""

import os
import json
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import InterFormerParquetDataset, get_interformer_eval_data
from model import InterFormer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Fallbacks matching train.py argparse defaults. These MUST match the defaults
# in train.py; otherwise the rebuilt model will shape-mismatch the saved
# state_dict when train_config.json is missing.
_FALLBACK_MODEL_CFG = {
    'embed_dim': 32,
    'n_layers': 2,
    'interaction': 'dcnv2',
    'n_heads': 4,
    'n_cls_tokens': 2,
    'n_pma_tokens': 1,
    'n_recent_tokens': 1,
    'n_sequences': 1,
    'seq_len': 100,
    'seq_vocab_size': 100000,
    'dropout': 0.1,
    'mlp_hidden_dims': '64,32',
}
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 0
_FALLBACK_SEQ_DOMAIN = 'domain_a'
_FALLBACK_MAX_DENSE_PER_FEAT = 32

# Keys used to build the model. Everything else in train_config.json is ignored.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (triggering fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"using hardcoded fallbacks. Shape mismatch may occur if training "
        f"used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from train_config; missing keys fall
    back to _FALLBACK_MODEL_CFG."""
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: InterFormerParquetDataset,
    model_cfg: Dict[str, Any],
    device: str = 'cpu',
) -> InterFormer:
    """Construct an InterFormer from the dataset schema + resolved model_cfg."""
    mlp_hidden_dims = [int(x.strip()) for x in model_cfg['mlp_hidden_dims'].split(',')]
    seq_vocab_size = model_cfg.get('seq_vocab_size', _FALLBACK_MODEL_CFG['seq_vocab_size'])

    logging.info(f"Building InterFormer with cfg: {model_cfg}")
    model = InterFormer(
        dense_dim=dataset.dense_dim,
        sparse_vocab_sizes=dataset.sparse_vocabs,
        seq_len=model_cfg['seq_len'],
        embed_dim=model_cfg['embed_dim'],
        n_layers=model_cfg['n_layers'],
        interaction=model_cfg['interaction'],
        n_heads=model_cfg['n_heads'],
        n_cls_tokens=model_cfg['n_cls_tokens'],
        n_pma_tokens=model_cfg['n_pma_tokens'],
        n_recent_tokens=model_cfg['n_recent_tokens'],
        n_sequences=model_cfg['n_sequences'],
        seq_vocab_size=seq_vocab_size,
        dropout=model_cfg['dropout'],
        mlp_hidden_dims=mlp_hidden_dims,
    ).to(device)
    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load state_dict; any missing/unexpected key fails fast."""
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path(model_dir: str) -> Optional[str]:
    """Locate the first ``*.pt`` file in MODEL_OUTPUT_PATH."""
    if not model_dir or not os.path.isdir(model_dir):
        return None
    for item in os.listdir(model_dir):
        if item.endswith('.pt'):
            return os.path.join(model_dir, item)
    return None


def main() -> None:
    # ---- Environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    if not model_dir:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    if not data_dir:
        raise ValueError("EVAL_DATA_PATH is not set")
    if not result_dir:
        raise ValueError("EVAL_RESULT_PATH is not set")

    os.makedirs(result_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer model_dir copy (exactly matches training);
    #      fall back to data_dir if missing ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # ---- Resolve hyperparams ----
    model_cfg = resolve_model_cfg(train_config)
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))
    seq_domain = train_config.get('seq_domain', _FALLBACK_SEQ_DOMAIN)
    max_dense_per_feat = int(train_config.get('max_dense_per_feat', _FALLBACK_MAX_DENSE_PER_FEAT))

    # ---- Create dataset ----
    test_loader, test_dataset = get_interformer_eval_data(
        data_dir=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seq_len=model_cfg['seq_len'],
        seq_domain=seq_domain,
        seq_vocab_size=model_cfg['seq_vocab_size'],
        max_dense_per_feat=max_dense_per_feat,
    )
    logging.info(f"Total test samples: {test_dataset.num_rows}")

    # ---- Build model: every structural hyperparam resolved from train_config ----
    model = build_model(test_dataset, model_cfg, device=device)

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path(model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"Directory contents: {os.listdir(model_dir) if os.path.isdir(model_dir) else 'N/A'}")
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    # ---- Inference ----
    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            dense = batch['dense'].to(device, non_blocking=True)
            sparse_ids = batch['sparse_ids'].to(device, non_blocking=True)
            seq_ids = batch['seq_ids'].to(device, non_blocking=True)
            seq_padding_mask = batch['seq_padding_mask'].to(device, non_blocking=True)
            user_ids = batch['user_id']

            logits = model(dense, sparse_ids, seq_ids, seq_padding_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    # ---- Save predictions.json ----
    predictions = {"predictions": dict(zip(all_user_ids, all_probs))}
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()