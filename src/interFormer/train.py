"""InterFormer training entry point.

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH       Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH       Checkpoint output directory
    TRAIN_LOG_PATH        Log directory
    TRAIN_TF_EVENTS_PATH  TensorBoard events directory (optional)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch

from utils import set_seed, EarlyStopping, create_logger, count_parameters
from dataset import InterFormerParquetDataset, get_interformer_data
from model import InterFormer
from trainer import InterFormerTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InterFormer Training")

    # ---- Paths (environment variables take precedence) ----
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # ---- Training hyperparameters ----
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=999)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Data pipeline ----
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of Row Groups used for training')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps (0 = epoch-level only)')
    parser.add_argument('--seq_len', type=int, default=500,
                        help='Padded sequence length（云端若 OOM 请调小，如 200）')
    parser.add_argument('--seq_vocab_size', type=int, default=100000,
                        help='Fallback hash bucket size for sequence item IDs')
    parser.add_argument('--max_dense_per_feat', type=int, default=0,
                        help='Max dim per dense feature (downsample, 0=use schema dim)')
    parser.add_argument('--item_id_vocab_size', type=int, default=100000,
                        help='Vocab size for item_id hashing')
    parser.add_argument('--emb_skip_threshold', type=int, default=500000,
                        help='Cap vocab size for embeddings (0=disabled). '
                             'Prevents OOM from high-cardinality features. '
                             'Matches PCVR emb_skip_threshold.')

    # ---- Model hyperparameters ----
    parser.add_argument('--embed_dim', type=int, default=32,
                        help='Embedding dimension d')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of InterFormer layers')
    parser.add_argument('--interaction', type=str, default='dcnv2',
                        choices=['fm', 'dcnv2', 'dhen'])
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_cls_tokens', type=int, default=2)
    parser.add_argument('--n_pma_tokens', type=int, default=1)
    parser.add_argument('--n_recent_tokens', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mlp_hidden_dims', type=str, default='64,32',
                        help='Comma-separated hidden dims for prediction head MLP')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH', None)

    # Local fallback defaults (when neither env var nor CLI flag is provided).
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir is None:
        args.data_dir = os.path.join(_script_dir, '..', 'data')
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(_script_dir, '..', 'checkpoints')
    if args.log_dir is None:
        args.log_dir = os.path.join(_script_dir, '..', 'logs')

    return args


def build_model(args, dataset: InterFormerParquetDataset) -> InterFormer:
    """Construct an InterFormer model from args + dataset metadata."""
    mlp_hidden_dims = [int(x.strip()) for x in args.mlp_hidden_dims.split(',')]
    model = InterFormer(
        dense_dim=dataset.dense_dim,
        sparse_vocab_sizes=dataset.sparse_vocabs,
        seq_len=args.seq_len,
        seq_vocab_sizes=dataset.seq_vocab_sizes,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        interaction=args.interaction,
        n_heads=args.n_heads,
        n_cls_tokens=args.n_cls_tokens,
        n_pma_tokens=args.n_pma_tokens,
        n_recent_tokens=args.n_recent_tokens,
        n_sequences=dataset.n_sequences,
        sparse_is_array=dataset.sparse_is_array,
        sparse_multi_dim=dataset.sparse_multi_dim,
        emb_skip_threshold=args.emb_skip_threshold,
        emb_skip_threshold=emb_skip_threshold,
        dropout=args.dropout,
        mlp_hidden_dims=mlp_hidden_dims,
    )
    return model


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Logger.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    # TensorBoard (optional).
    if args.tf_events_dir:
        Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tf_events_dir)
    else:
        writer = None

    # ---- Data loading (streaming RowGroup-based, PCVRHyFormer pattern) ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    logging.info(f"Loading data from {args.data_dir}")
    logging.info(f"Schema: {schema_path}")

    train_loader, val_loader, train_ds = get_interformer_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_len=args.seq_len,
        max_dense_per_feat=args.max_dense_per_feat,
        seq_vocab_size=args.seq_vocab_size,
        item_id_vocab_size=args.item_id_vocab_size,
        emb_skip_threshold=args.emb_skip_threshold,
    )

    # ---- Build model ----
    model = build_model(args, train_ds).to(args.device)

    total_params = count_parameters(model)
    logging.info(f"InterFormer model created: embed_dim={args.embed_dim}, "
                 f"n_layers={args.n_layers}, interaction={args.interaction}, "
                 f"n_sequences={train_ds.n_sequences}")
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.n_layers,
        "head": args.n_heads,
        "hidden": args.embed_dim,
    }

    trainer = InterFormerTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        weight_decay=args.weight_decay,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
    )

    trainer.train()
    if writer is not None:
        writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()