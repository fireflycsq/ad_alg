"""OneTrans training entry point.

Usage:
    python train.py --data_dir src/data --schema_path src/data/schema.json \\
                    --ckpt_dir src/checkpoints/one_trans --log_dir src/logs/one_trans
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import OneTrans
from trainer import OneTransTrainer


def build_feature_specs(schema: FeatureSchema, per_position_vocab_sizes: List[int]) -> List[Tuple[int, int, int]]:
    specs = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OneTrans Training")

    # Paths
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--schema_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Data pipeline
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--buffer_batches', type=int, default=5)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--eval_every_n_steps', type=int, default=0)

    # Model (OneTransS defaults from paper Table 2)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_ns_tokens', type=int, default=12)
    parser.add_argument('--hidden_mult', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--emb_skip_threshold', type=int, default=0)

    # Loss
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.1)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # Optimizer
    parser.add_argument('--sparse_lr', type=float, default=0.01)
    parser.add_argument('--grad_clip_dense', type=float, default=90.0)
    parser.add_argument('--grad_clip_sparse', type=float, default=120.0)

    # Sequence max lengths
    parser.add_argument('--seq_max_lens', type=str,
                        default='domain_a:50,domain_b:50,domain_c:50,domain_d:50')

    args = parser.parse_args()

    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)

    return args


def main() -> None:
    args = parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    # ── Data loading ──
    schema_path = args.schema_path or os.path.join(args.data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())

    logging.info("Loading data...")
    train_loader, valid_loader, dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    # ── Build model ──
    user_int_specs = build_feature_specs(dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_specs = build_feature_specs(dataset.item_int_schema, dataset.item_int_vocab_sizes)

    model = OneTrans(
        user_int_feature_specs=user_int_specs,
        item_int_feature_specs=item_int_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_ns_tokens=args.num_ns_tokens,
        hidden_mult=args.hidden_mult,
        dropout=args.dropout,
        emb_skip_threshold=args.emb_skip_threshold,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    sparse_params = sum(p.numel() for p in model.get_sparse_params())
    dense_params = sum(p.numel() for p in model.get_dense_params())
    logging.info(f"OneTrans model created: {total_params:,} total params "
                 f"(sparse={sparse_params:,}, dense={dense_params:,})")
    logging.info(f"Seq domains: {dataset.seq_domains}")

    # ── Training ──
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience, label='model')

    ckpt_params = {
        "layer": args.num_layers,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = OneTransTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        grad_clip_dense=args.grad_clip_dense,
        grad_clip_sparse=args.grad_clip_sparse,
        ckpt_params=ckpt_params,
        schema_path=schema_path,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
    )

    trainer.train()
    logging.info("Training complete!")


if __name__ == "__main__":
    main()