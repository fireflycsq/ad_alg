"""InterFormer eval dataset — reads raw multi-column Parquet for inference.

Uses the same ``schema.json`` format as training data:
{
  "user_int": [[fid, vocab_size, dim], ...],
  "item_int": [],
  "user_dense": [[fid, dim], ...],
  "item_dense": [],
  "seq": {
    "domain_a": {
      "prefix": "seq_domain_a",
      "ts_fid": 100,
      "features": [[100, vocab_size], ...]
    }
  }
}

Column naming convention:
  - user_dense_feats_{fid}   (list<float>)
  - user_int_feats_{fid}     (int or list<int>)
  - {seq_prefix}_{fid}       (list<int64>)
  - user_id, timestamp, label_type
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
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

torch.multiprocessing.set_sharing_strategy('file_system')


class InterFormerParquetDataset(IterableDataset):
    """IterableDataset reading raw multi-column Parquet for InterFormer.

    Parses the full PCVRHyFormer schema, extracting the fields InterFormer
    requires: dense features (user_dense), sparse IDs (user_int), one
    behaviour sequence, and binary labels.

    Exposes ``dense_dim`` and ``sparse_vocabs`` so the training script can
    construct the model with the correct input dimensions.

    When ``is_training=False``, labels are not extracted and ``user_id`` is
    included in each batch dict for inference.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_len: int = 100,
        seq_domain: str = 'domain_a',
        seq_vocab_size: int = 100000,
        max_dense_per_feat: int = 32,
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
        self.seq_domain = seq_domain
        self.seq_vocab_size = seq_vocab_size
        self.max_dense_per_feat = max_dense_per_feat
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
        self._buf_dense = np.zeros((B, self.dense_dim), dtype=np.float32)
        self._buf_sparse = np.zeros((B, len(self.sparse_vocabs)), dtype=np.int64)
        self._buf_seq = np.zeros((B, seq_len), dtype=np.int64)
        self._buf_seq_mask = np.zeros((B, seq_len), dtype=np.bool_)

        # Pre-compute column plans.
        self._dense_plan = []
        offset = 0
        for fid, raw_dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            if ci is None:
                continue
            use_dim = min(raw_dim, max_dense_per_feat) if max_dense_per_feat > 0 else raw_dim
            self._dense_plan.append((ci, raw_dim, use_dim, offset))
            offset += use_dim

        self._sparse_plan = []
        for i, (fid, vs, dim) in enumerate(self._user_int_cols):
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            if ci is None:
                continue
            self._sparse_plan.append((ci, dim, i, vs))

        prefix = self._seq_prefix
        if self._seq_features:
            first_fid, first_vs = self._seq_features[0]
            self._seq_col_idx = self._col_idx.get(f'{prefix}_{first_fid}')
        else:
            self._seq_col_idx = None

        logging.info(
            f"InterFormerParquetDataset: {self.num_rows} rows, "
            f"dense_dim={self.dense_dim}, n_sparse={len(self.sparse_vocabs)}, "
            f"seq_len={seq_len}, is_training={is_training}")

    def _load_schema(self, schema_path: str) -> None:
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self._user_dense_cols: List[List[int]] = raw.get('user_dense', [])
        self.dense_dim = 0
        for fid, dim in self._user_dense_cols:
            if self.max_dense_per_feat > 0 and dim > self.max_dense_per_feat:
                self.dense_dim += self.max_dense_per_feat
            else:
                self.dense_dim += dim

        self._user_int_cols: List[List[int]] = raw.get('user_int', [])
        self.sparse_vocabs: List[int] = [vs for _, vs, _ in self._user_int_cols]

        seq_cfg = raw.get('seq', {})
        domain = self.seq_domain
        if domain not in seq_cfg:
            raise KeyError(
                f"seq_domain='{domain}' not found in schema. "
                f"Available: {list(seq_cfg.keys())}")
        self._seq_prefix: str = seq_cfg[domain]['prefix']
        self._seq_features: List[List[int]] = seq_cfg[domain]['features']

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

    # ---- per-column helpers -------------------------------------------------

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
            ul = min(end - start, raw_dim, use_dim)
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

    # ---- batch conversion ---------------------------------------------------

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        B = batch.num_rows

        # ---- user_id ----
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        # ---- label ----
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)

        # ---- Dense features (user_dense_feats_{fid}) ----
        dense = self._buf_dense[:B]
        dense[:] = 0
        for ci, raw_dim, use_dim, offset in self._dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float(col, raw_dim, use_dim, B)
            dense[:, offset:offset + use_dim] = padded

        # ---- Sparse features (user_int_feats_{fid}) ----
        sparse = self._buf_sparse[:B]
        sparse[:] = 0
        for ci, dim, slot, vs in self._sparse_plan:
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

        # ---- Sequence features ({prefix}_{fid}) ----
        seq = self._buf_seq[:B]
        seq[:] = 0
        seq_mask = self._buf_seq_mask[:B]
        seq_mask[:] = True

        if self._seq_col_idx is not None:
            col = batch.column(self._seq_col_idx)
            padded, lengths = self._pad_varlen_int(col, self.seq_len, B)
            padded[padded < 0] = 0
            padded = padded % self.seq_vocab_size
            seq[:] = padded
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
        if self.is_training:
            result['label'] = torch.from_numpy(labels)
        return result


def get_interformer_eval_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    num_workers: int = 0,
    seq_len: int = 100,
    seq_domain: str = 'domain_a',
    seq_vocab_size: int = 100000,
    max_dense_per_feat: int = 32,
) -> Tuple[DataLoader, InterFormerParquetDataset]:
    """Create a DataLoader and dataset for InterFormer inference.

    Returns:
        (loader, dataset) — dataset exposes ``dense_dim`` and
        ``sparse_vocabs`` for model construction.
    """
    dataset = InterFormerParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_len=seq_len,
        seq_domain=seq_domain,
        seq_vocab_size=seq_vocab_size,
        max_dense_per_feat=max_dense_per_feat,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )
    return loader, dataset