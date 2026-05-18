#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# InterFormer local test config:
#   - seq_len=5000 to cover full sequence lengths (max ~3951 in demo data)
#   - embed_dim=64 matches PCVRHyFormer d_model
#   - DHEN interaction: stronger than FM, fewer params than DCNv2 (balanced)
#   - 3 layers for deeper interaction
#   - num_workers=0 for local testing
python3 -u "${SCRIPT_DIR}/train.py" \
    --embed_dim 64 \
    --n_layers 3 \
    --n_heads 8 \
    --interaction dhen \
    --n_cls_tokens 4 \
    --n_pma_tokens 2 \
    --n_recent_tokens 2 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_epochs 999 \
    --patience 5 \
    --dropout 0.01 \
    --mlp_hidden_dims '256,128' \
    --seq_len 5000 \
    --num_workers 0 \
    "$@"