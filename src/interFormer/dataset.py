"""Dataset processing for InterFormer."""

import torch
from torch.utils.data import Dataset, DataLoader


class CTRDataset(Dataset):
    """CTR dataset for InterFormer."""
    
    def __init__(self, dense, sparse_ids, seq_ids, labels, seq_padding_mask=None):
        """
        Args:
            dense: (N, dense_dim) tensor of dense features
            sparse_ids: (N, n_sparse) tensor of sparse feature indices
            seq_ids: (N, T) or (N, k, T) tensor of sequence ids
            labels: (N,) tensor of labels
            seq_padding_mask: (N, T) tensor of padding masks, optional
        """
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
                self.dense[idx],
                self.sparse_ids[idx],
                self.seq_ids[idx],
                self.seq_padding_mask[idx],
                self.labels[idx]
            )
        else:
            return (
                self.dense[idx],
                self.sparse_ids[idx],
                self.seq_ids[idx],
                self.labels[idx]
            )


def make_synthetic_batch(B: int, dense_dim: int, n_sparse: int,
                         vocab_size: int, seq_len: int, device: str = "cpu"):
    """Generate a random batch for quick testing."""
    dense = torch.randn(B, dense_dim, device=device)
    sparse_cols = [torch.randint(0, vs, (B,), device=device) for vs in [100, 200, 150, 300][:n_sparse]]
    sparse_ids = torch.stack(sparse_cols, dim=1)
    seq_ids = torch.randint(1, min(sparse_vocab_sizes := [100, 200, 150, 300][:n_sparse]) if n_sparse else vocab_size, (B, seq_len), device=device)
    # Random padding mask: last 20% of sequence is padding
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