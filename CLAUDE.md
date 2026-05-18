# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains PyTorch implementations of deep learning models for click-through rate (CTR) and post-click conversion rate (PCVR) prediction:

- **PCVRHyFormer** (`src/PCVRHyFormer/`): A hybrid transformer model that processes multiple user/item behavior sequences with non-sequence (NS) features for PCVR prediction.
- **InterFormer** (`src/interFormer/`): A model for CTR prediction using heterogeneous feature interaction learning with three core architectures: Interaction Arch, Sequence Arch, and Cross Arch.

## Commands

### Training PCVRHyFormer
```bash
cd src/PCVRHyFormer
python train.py --data_dir <path> --ckpt_dir <path> --log_dir <path>
```

Environment variables override CLI flags:
- `TRAIN_DATA_PATH`: Training data directory (parquet files + schema.json)
- `TRAIN_CKPT_PATH`: Checkpoint output directory
- `TRAIN_LOG_PATH`: Log directory

Key hyperparameters:
- `--num_epochs 10`: Maximum training epochs
- `--batch_size 256`: Batch size
- `--lr 1e-4`: Dense optimizer learning rate (AdamW)
- `--sparse_lr 0.05`: Sparse embedding learning rate (Adagrad)
- `--num_hyformer_blocks 2`: Number of MultiSeqHyFormerBlock layers
- `--d_model 64`: Hidden dimension (must be divisible by T = num_queries*num_sequences + num_ns)
- `--seq_encoder_type transformer`: Sequence encoder variant (`swiglu`, `transformer`, `longer`)
- `--rank_mixer_mode full`: RankMixer mode (`full`, `ffn_only`, `none`)
- `--ns_tokenizer_type rankmixer`: NS tokenizer variant (`group`, `rankmixer`)
- `--loss_type bce`: Loss function (`bce`, `focal`)

### Training InterFormer (synthetic data demo)
```bash
cd src/interFormer
python train.py
```

The InterFormer training script uses synthetic data for demonstration. To use real data, modify the data loading in `train.py`.

### Running InterFormer standalone test
```bash
cd src/interFormer
python interformer.py
```

## Architecture Overview

### PCVRHyFormer (src/PCVRHyFormer/)
- **model.py**: Main model with components:
  - `PCVRHyFormer`: Main model class combining NS tokenizers, sequence embeddings, and HyFormer blocks
  - `MultiSeqHyFormerBlock`: Multi-sequence processing block with sequence evolution, query decoding, and query boosting
  - `RankMixerBlock`: Query boosting via token mixing + FFN
  - `MultiSeqQueryGenerator`: Generates per-sequence query tokens from NS + sequence pooling
  - `GroupNSTokenizer`: Projects grouped discrete features to single NS tokens (one per group)
  - `RankMixerNSTokenizer`: Concatenates all embeddings, splits into configurable number of NS tokens
  - `RotaryEmbedding` + `RoPEMultiheadAttention`: Rotary position encoding support
  - Sequence encoders: `SwiGLUEncoder`, `TransformerEncoder`, `LongerEncoder` (Top-K compressed)
- **dataset.py**: Parquet dataset loader with feature schema, time bucketing, and shuffle buffer
- **trainer.py**: Dual optimizer (Adagrad for sparse, AdamW for dense), early stopping, checkpoint sidecar files
- **train.py**: Entry point with CLI arguments, schema loading, NS group configuration
- **utils.py**: Early stopping, logging, focal loss, seed setting

Data input format (via `ModelInput` NamedTuple):
- `user_int_feats`, `item_int_feats`: Integer features for NS tokenizers
- `user_dense_feats`, `item_dense_feats`: Dense float features (projected to NS tokens)
- `seq_data`: Dict mapping domain names (seq_a, seq_b, seq_c, seq_d) to tensors [B, S, L]
- `seq_lens`: Actual sequence lengths per domain
- `seq_time_buckets`: Time-delta bucket ids for temporal encoding

### InterFormer (src/interFormer/)
- **interformer.py**: Complete implementation including:
  - `InterFormer`: Main model with stacked Interaction/Sequence/Cross Arch layers
  - `InteractionArch`: Feature interaction using FM, DCNv2, or DHEN modules
  - `SequenceArch`: PFFN (Personalized FFN) + self/cross attention
  - `CrossArch`: PMA-based summary exchange between non-sequence and sequence sides
  - `PMA`: Pooling by Multi-Head Attention for sequence summarization
  - `CTRTrainer`: Simple training loop with BCEWithLogitsLoss
- **model.py**: Duplicate of interformer.py (same classes)
- **dataset.py**: Simple CTR dataset wrapper, synthetic data generation
- **trainer.py**: Training utilities
- **train.py**: Synthetic demo training script

## Key Data Format

### PCVRHyFormer Parquet Data
Data directory must contain:
- `*.parquet` files with columns: `user_int_feats_{fid}`, `item_int_feats_{fid}`, `user_dense_feats_{fid}`, `seq_{domain}_{fid}`, `timestamp`, `label_type`, `user_id`
- `schema.json` defining feature layouts:
  ```json
  {
    "user_int": [[fid, vocab_size, dim], ...],
    "item_int": [[fid, vocab_size, dim], ...],
    "user_dense": [[fid, dim], ...],
    "seq": {
      "seq_a": {"prefix": "seq_a", "ts_fid": fid, "features": [[fid, vocab_size], ...]},
      ...
    }
  }
  ```

### NS Groups Configuration
`ns_groups.json` defines grouping for discrete features:
```json
{
  "user_ns_groups": {"U1": [fid1, fid2], ...},
  "item_ns_groups": {"I1": [fid1, fid2], ...}
}
```
If missing, each feature becomes a singleton group.

## Important Constraints

- `d_model` must be divisible by `T = num_queries * num_sequences + num_ns` when `rank_mixer_mode='full'`
- Time bucket count is fixed by `NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1` in dataset.py
- Embedding vocab sizes in schema.json must match actual data (or use `clip_vocab=True`)
- High-cardinality embeddings can be skipped via `--emb_skip_threshold` to save GPU memory