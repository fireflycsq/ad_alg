"""Dataset processing for InterFormer — synthetic data + Parquet streaming.

Loads ALL features per the InterFormer paper (Section 4.1):
  - Non-sequence: user_int + item_int + user_dense + item_dense
  - Sequence: ALL sequence domains (k sequences), fused via MaskNet
"""

import os
import json
import logging
import random
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Any, Dict, Iterator, List, Optional, Tuple

torch.multiprocessing.set_sharing_strategy('file_system')


class CTRDataset(Dataset):
    """CTR dataset for InterFormer."""

    def __init__(self, dense, sparse_ids, seq_ids, labels, seq_padding_mask=None):
        self.dense = dense
        self.sparse_ids = sparse_ids
        self.seq_ids = seq_ids
        self.labels = labels
        self.seq_padding_mask = seq_padding_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.seq_padding_mask is not None:
            return (
                self.dense[idx], self.sparse_ids[idx], self.seq_ids[idx],
                self.seq_padding_mask[idx], self.labels[idx]
            )
        else:
            return (
                self.dense[idx], self.sparse_ids[idx], self.seq_ids[idx],
                self.labels[idx]
            )


def make_synthetic_batch(B: int, dense_dim: int, n_sparse: int,
                         vocab_size: int, seq_len: int, device: str = "cpu"):
    """Generate a random batch for quick testing."""
    dense = torch.randn(B, dense_dim, device=device)
    sparse_vocab_sizes = [100, 200, 150, 300][:n_sparse]
    sparse_cols = [torch.randint(0, vs, (B,), device=device) for vs in sparse_vocab_sizes]
    sparse_ids = torch.stack(sparse_cols, dim=1)
    seq_ids = torch.randint(1, vocab_size, (B, seq_len), device=device)
    pad_start = int(seq_len * 0.8)
    seq_padding_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    seq_padding_mask[:, pad_start:] = True
    labels = torch.randint(0, 2, (B,), device=device)
    return dense, sparse_ids, seq_ids, seq_padding_mask, labels


def create_dataloaders(train_data, val_data, batch_size=64):
    """Create dataloaders for training and validation."""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Parquet dataset
# ---------------------------------------------------------------------------

class InterFormerParquetDataset(IterableDataset):
    """IterableDataset reading multi-column Parquet for InterFormer training.

    Loads ALL features defined in the schema:
      - user_int  → sparse user features
      - item_int  → sparse item features
      - user_dense → dense user features
      - item_dense → dense item features (if present)
      - seq       → ALL sequence domains (k sequences fused via MaskNet)

    Exposes metadata so the training script can construct the model:
      - dense_dim, sparse_vocabs (combined user+item)
      - n_sequences, seq_vocab_sizes (per domain)
      - seq_len
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_len: int = 500,
        seq_vocab_size: int = 100000,
        max_dense_per_feat: int = 0,
        item_id_vocab_size: int = 100000,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        is_training: bool = True,
    ) -> None:
        super().__init__()

        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_vocab_size = seq_vocab_size
        self.max_dense_per_feat = max_dense_per_feat
        self.item_id_vocab_size = item_id_vocab_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.is_training = is_training

        # Build Row Group list with optional range filtering.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.
        self._load_schema(schema_path)

        # Pre-compute column index lookup.
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # Pre-allocate numpy buffers.
        B = batch_size
        n_user_sparse = len(self.user_sparse_vocabs)
        n_item_sparse = len(self.item_sparse_vocabs)
        n_sparse_total = n_user_sparse + n_item_sparse
        n_seqs = self.n_sequences
        max_sf = self.max_seq_features

        self._buf_dense = np.zeros((B, self.dense_dim), dtype=np.float32)
        self._buf_sparse = np.zeros((B, n_sparse_total + 1), dtype=np.int64)  # +1 for item_id
        self._buf_seq = np.zeros((B, n_seqs, max_sf, seq_len), dtype=np.int64)
        self._buf_seq_mask = np.zeros((B, seq_len), dtype=np.bool_)

        # ---- Sparse array feature buffers (dim > 1) ----
        self.n_array_feats = sum(1 for b in self.sparse_is_array if b)
        self.max_array_dim = max(self.sparse_multi_dim) if self.n_array_feats > 0 else 0
        self._buf_sparse_multi = np.zeros(
            (B, self.n_array_feats, self.max_array_dim), dtype=np.int64,
        ) if self.n_array_feats > 0 else None
        self._buf_sparse_multi_mask = np.zeros(
            (B, self.n_array_feats, self.max_array_dim), dtype=np.bool_,
        ) if self.n_array_feats > 0 else None

        # ---- Dense plans ----
        self._dense_plan = []
        offset = 0
        for fid, raw_dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            if ci is None:
                continue
            use_dim = min(raw_dim, max_dense_per_feat) if max_dense_per_feat > 0 else raw_dim
            self._dense_plan.append((ci, raw_dim, use_dim, offset, 'user'))
            offset += use_dim
        for fid, raw_dim in self._item_dense_cols:
            ci = self._col_idx.get(f'item_dense_feats_{fid}')
            if ci is None:
                continue
            use_dim = min(raw_dim, max_dense_per_feat) if max_dense_per_feat > 0 else raw_dim
            self._dense_plan.append((ci, raw_dim, use_dim, offset, 'item'))
            offset += use_dim

        # ---- Sparse plans ----
        self._sparse_plan = []
        _array_idx = 0
        # item_id is slot 0 (prepended) — always scalar
        self._sparse_plan.append({
            'col_name': 'item_id',
            'dim': 1,
            'slot': 0,
            'vocab_size': self.item_id_vocab_size,
            'is_item_id': True,
            'is_array': False,
            'array_idx': -1,
        })
        for i, (fid, vs, dim) in enumerate(self._user_int_cols):
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            if ci is None:
                continue
            is_arr = dim > 1
            self._sparse_plan.append({
                'col_idx': ci, 'dim': dim,
                'slot': 1 + i,
                'vocab_size': vs, 'is_item_id': False,
                'is_array': is_arr,
                'array_idx': _array_idx if is_arr else -1,
            })
            if is_arr:
                _array_idx += 1
        for i, (fid, vs, dim) in enumerate(self._item_int_cols):
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            if ci is None:
                continue
            is_arr = dim > 1
            self._sparse_plan.append({
                'col_idx': ci, 'dim': dim,
                'slot': 1 + n_user_sparse + i,
                'vocab_size': vs, 'is_item_id': False,
                'is_array': is_arr,
                'array_idx': _array_idx if is_arr else -1,
            })
            if is_arr:
                _array_idx += 1

        # ---- Sequence plans: ALL features per domain ----
        self._seq_plans = []
        for domain in self.seq_domains:
            seq_cfg = self._seq_cfg[domain]
            prefix = seq_cfg['prefix']
            features = seq_cfg['features']
            feat_plans = []
            for fid, vs in features:
                ci = self._col_idx.get(f'{prefix}_{fid}')
                feat_plans.append({'fid': fid, 'col_idx': ci, 'vocab_size': vs})
            self._seq_plans.append({
                'domain': domain,
                'prefix': prefix,
                'features': feat_plans,
            })

        logging.info(
            f"InterFormerParquetDataset: {self.num_rows} rows, "
            f"dense_dim={self.dense_dim}, "
            f"n_user_sparse={n_user_sparse}, n_item_sparse={n_item_sparse}, "
            f"n_seqs={n_seqs} ({', '.join(self.seq_domains)}), "
            f"max_seq_features={max_sf}, seq_len={seq_len}, shuffle={shuffle}")

    # ---- Schema loading ---------------------------------------------------

    def _load_schema(self, schema_path: str) -> None:
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # User dense
        self._user_dense_cols: List[List[int]] = raw.get('user_dense', [])
        self.user_dense_dim = 0
        for fid, dim in self._user_dense_cols:
            use_dim = min(dim, self.max_dense_per_feat) if self.max_dense_per_feat > 0 else dim
            self.user_dense_dim += use_dim

        # Item dense (may be absent)
        self._item_dense_cols: List[List[int]] = raw.get('item_dense', [])
        self.item_dense_dim = 0
        for fid, dim in self._item_dense_cols:
            use_dim = min(dim, self.max_dense_per_feat) if self.max_dense_per_feat > 0 else dim
            self.item_dense_dim += use_dim

        self.dense_dim = self.user_dense_dim + self.item_dense_dim

        # User int (sparse)
        self._user_int_cols: List[List[int]] = raw.get('user_int', [])
        self.user_sparse_vocabs: List[int] = [vs for _, vs, _ in self._user_int_cols]

        # Item int (sparse)
        self._item_int_cols: List[List[int]] = raw.get('item_int', [])
        self.item_sparse_vocabs: List[int] = [vs for _, vs, _ in self._item_int_cols]

        # Combined sparse vocabs (for model construction)
        # item_id is prepended as the first sparse feature
        self.sparse_vocabs: List[int] = (
            [self.item_id_vocab_size] +
            self.user_sparse_vocabs +
            self.item_sparse_vocabs
        )

        # Per-slot array metadata (same order as sparse_vocabs)
        self.sparse_is_array: List[bool] = (
            [False] +  # item_id is scalar
            [dim > 1 for _, _, dim in self._user_int_cols] +
            [dim > 1 for _, _, dim in self._item_int_cols]
        )
        self.sparse_multi_dim: List[int] = (
            [0] +  # item_id
            [dim if dim > 1 else 0 for _, _, dim in self._user_int_cols] +
            [dim if dim > 1 else 0 for _, _, dim in self._item_int_cols]
        )

        # All sequence domains
        seq_cfg = raw.get('seq', {})
        self._seq_cfg = seq_cfg
        self.seq_domains: List[str] = sorted(seq_cfg.keys())
        self.n_sequences: int = len(self.seq_domains)

        # Per-domain per-feature vocab sizes: seq_vocab_sizes[domain_idx][feat_idx]
        self.seq_vocab_sizes: List[List[int]] = []
        self.seq_features_per_domain: List[int] = []
        for domain in self.seq_domains:
            features = seq_cfg[domain]['features']
            self.seq_features_per_domain.append(len(features))
            self.seq_vocab_sizes.append([vs for _, vs in features])

        self.max_seq_features: int = max(self.seq_features_per_domain) if self.seq_features_per_domain else 0

    # ---- Length -----------------------------------------------------------

    def __len__(self) -> int:
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(self, buffer: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged['dense'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Per-column helpers -----------------------------------------------

    @staticmethod
    def _pad_varlen_float(
        arrow_col: "pa.ListArray", raw_dim: int, use_dim: int, B: int
    ) -> np.ndarray:
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        padded = np.zeros((B, use_dim), dtype=np.float32)
        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            if end <= start:
                continue
            ul = min(end - start, use_dim)
            padded[i, :ul] = values[start:start + ul]
        return padded

    @staticmethod
    def _pad_varlen_int(
        arrow_col: "pa.ListArray", max_len: int, B: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)
        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            rl = end - start
            if rl <= 0:
                continue
            ul = min(rl, max_len)
            padded[i, :ul] = values[start:start + ul]
            lengths[i] = ul
        padded[padded <= 0] = 0
        return padded, lengths

    # ---- Batch conversion -------------------------------------------------

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        B = batch.num_rows

        # ---- user_id ----
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- label ----
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)

        # ---- Dense features (user_dense_feats_{fid} + item_dense_feats_{fid}) ----
        dense = self._buf_dense[:B]
        dense[:] = 0
        for ci, raw_dim, use_dim, offset, _kind in self._dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float(col, raw_dim, use_dim, B)
            dense[:, offset:offset + use_dim] = padded

        # ---- Sparse features (item_id + user_int_feats_{fid} + item_int_feats_{fid}) ----
        sparse = self._buf_sparse[:B]
        sparse[:] = 0
        if self._buf_sparse_multi is not None:
            sparse_multi = self._buf_sparse_multi[:B]
            sparse_multi[:] = 0
            sparse_multi_mask = self._buf_sparse_multi_mask[:B]
            sparse_multi_mask[:] = False
        else:
            sparse_multi = None
            sparse_multi_mask = None

        for plan in self._sparse_plan:
            if plan.get('is_item_id'):
                # item_id: scalar int64, high cardinality → hash/mod
                col = batch.column(self._col_idx['item_id'])
                arr = col.to_numpy(zero_copy_only=False).astype(np.int64)
                arr = arr % plan['vocab_size']
                arr[arr < 0] = 0
                sparse[:, plan['slot']] = arr
                continue

            ci = plan['col_idx']
            dim = plan['dim']
            slot = plan['slot']
            vs = plan['vocab_size']
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                arr[arr >= vs] = 0
                sparse[:, slot] = arr
            else:
                padded, _ = self._pad_varlen_int(col, dim, B)
                padded[padded >= vs] = 0
                sparse[:, slot] = padded[:, 0]
                # Store full array for mean-pooling in model
                if plan['is_array'] and sparse_multi is not None:
                    aidx = plan['array_idx']
                    w = min(dim, self.max_array_dim)
                    sparse_multi[:, aidx, :w] = padded[:, :w]
                    sparse_multi_mask[:, aidx, :w] = (padded[:, :w] != 0)

        # ---- Sequence features: ALL domains, ALL features per domain ----
        n_seqs = self.n_sequences
        max_sf = self.max_seq_features
        seq = self._buf_seq[:B]
        seq[:] = 0
        seq_mask = self._buf_seq_mask[:B]
        seq_mask[:] = True

        for k, plan in enumerate(self._seq_plans):
            for f, feat in enumerate(plan['features']):
                if feat['col_idx'] is not None:
                    col = batch.column(feat['col_idx'])
                    padded, lengths = self._pad_varlen_int(col, self.seq_len, B)
                    padded[padded < 0] = 0
                    if feat['vocab_size'] > 0:
                        padded = padded % feat['vocab_size']
                    seq[:, k, f, :] = padded
                    # Use first domain's first feature's lengths for mask
                    if k == 0 and f == 0:
                        for i in range(B):
                            if lengths[i] > 0:
                                seq_mask[i, :lengths[i]] = False

        result = {
            'dense': torch.from_numpy(dense.copy()),
            'sparse_ids': torch.from_numpy(sparse.copy()),
            'seq_ids': torch.from_numpy(seq.copy()),
            'seq_padding_mask': torch.from_numpy(seq_mask.copy()),
            'user_id': user_ids,
        }
        if sparse_multi is not None:
            result['sparse_multi'] = torch.from_numpy(sparse_multi.copy())
            result['sparse_multi_mask'] = torch.from_numpy(sparse_multi_mask.copy())
        if self.is_training:
            result['label'] = torch.from_numpy(labels)
        return result


def get_interformer_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    buffer_batches: int = 20,
    seed: int = 42,
    seq_len: int = 500,
    max_dense_per_feat: int = 0,
    seq_vocab_size: int = 100000,
    item_id_vocab_size: int = 100000,
) -> Tuple[DataLoader, DataLoader, InterFormerParquetDataset]:
    """Create train / valid DataLoaders using Row Group split.

    The validation split is taken as the last ``(1 - train_ratio)`` fraction
    of Row Groups.

    Returns:
        (train_loader, valid_loader, train_dataset) — dataset exposes
        ``dense_dim``, ``sparse_vocabs``, ``n_sequences``, ``seq_vocab_sizes``
        for model construction.
    """
    random.seed(seed)
    import glob as _glob

    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))
    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_train_rgs = max(1, int(total_rgs * train_ratio))
    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{total_rgs - n_train_rgs} valid ({valid_rows} rows)")

    train_dataset = InterFormerParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_len=seq_len,
        seq_vocab_size=seq_vocab_size,
        max_dense_per_feat=max_dense_per_feat,
        item_id_vocab_size=item_id_vocab_size,
        shuffle=True,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        is_training=True,
    )

    valid_dataset = InterFormerParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_len=seq_len,
        seq_vocab_size=seq_vocab_size,
        max_dense_per_feat=max_dense_per_feat,
        item_id_vocab_size=item_id_vocab_size,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        is_training=True,
    )

    use_cuda = torch.cuda.is_available()
    train_kwargs = {}
    if num_workers > 0:
        train_kwargs['prefetch_factor'] = 2
    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **train_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"InterFormer Parquet train: {train_rows} rows, "
                 f"valid: {valid_rows} rows, batch_size={batch_size}")

    return train_loader, valid_loader, train_dataset