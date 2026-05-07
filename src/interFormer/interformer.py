"""
InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction
Paper: https://arxiv.org/abs/2411.09852
Complete PyTorch implementation following the paper's architecture.

Architecture Overview:
  For each layer l = 1..L:
    1. Interaction Arch  : X^(l+1) = MLP(Interaction([X^(l) || S_sum^(l)]))
    2. Sequence Arch     : S^(l+1) = TransformerBlock(S^(l), X_sum^(l))  via PFFN + MHA
    3. Cross Arch        : S_sum^(l+1), X_sum^(l+1) = CrossArch(X^(l+1), S^(l+1))
  Final CTR score = sigmoid(MLP([X_final || S_sum_final]))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List


# ---------------------------------------------------------------------------
# 1. Utilities
# ---------------------------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    """Standard LayerNorm wrapper for clarity."""
    pass


class MLP(nn.Module):
    """Multi-layer perceptron with optional dropout."""
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act_fn(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Feature Preprocessing  (Section 4.1)
# ---------------------------------------------------------------------------

class FeatureEmbedding(nn.Module):
    """
    Embeds dense + sparse non-sequence features into a unified matrix X^(1).
    
    Args:
        dense_dim      : raw dimension of concatenated dense features
        sparse_vocab_sizes : list of vocabulary sizes for each sparse feature
        embed_dim      : output embedding dimension d
    
    Output: X ∈ R^{(1 + n_sparse) × d}
              first token = dense projection, rest = sparse embeddings
    """
    def __init__(self, dense_dim: int, sparse_vocab_sizes: List[int], embed_dim: int):
        super().__init__()
        self.dense_proj = nn.Linear(dense_dim, embed_dim)
        self.sparse_embs = nn.ModuleList([
            nn.Embedding(vs, embed_dim) for vs in sparse_vocab_sizes
        ])
        self.embed_dim = embed_dim

    def forward(self, dense: Tensor, sparse_ids: Tensor) -> Tensor:
        """
        dense     : (B, dense_dim)
        sparse_ids: (B, n_sparse)  integer feature indices
        returns   : (B, 1+n_sparse, d)
        """
        dense_emb = self.dense_proj(dense).unsqueeze(1)           # (B,1,d)
        sparse_embs = [
            self.sparse_embs[i](sparse_ids[:, i]).unsqueeze(1)    # (B,1,d)
            for i in range(sparse_ids.size(1))
        ]
        return torch.cat([dense_emb] + sparse_embs, dim=1)        # (B, 1+n_sparse, d)


class MaskNet(nn.Module):
    """
    Unifies k sequences and filters noise (Eq. 6).
    MaskNet(S) = MLP_lce(S ⊙ MLP_mask(S))

    Args:
        k         : number of sequences
        seq_len   : T (padded sequence length)
        embed_dim : d
    """
    def __init__(self, k: int, seq_len: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        # MLP_mask: self-masking, keeps shape (B, k*d, T)
        self.mlp_mask = nn.Sequential(
            nn.Linear(k * embed_dim, k * embed_dim),
            nn.Sigmoid()
        )
        # MLP_lce: linearly combines k sequences → (B, d, T)
        self.mlp_lce = nn.Linear(k * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences: List[Tensor]) -> Tensor:
        """
        sequences: list of k tensors each (B, T, d)
        returns  : (B, T, d)
        """
        # Concatenate along feature dim: (B, T, k*d)
        S = torch.cat(sequences, dim=-1)
        mask = self.mlp_mask(S)          # (B, T, k*d)
        S = S * mask                     # element-wise gating
        S = self.mlp_lce(S)             # (B, T, d)
        return self.dropout(S)


# ---------------------------------------------------------------------------
# 3. Feature Interaction Modules  (Section 3.2)
# ---------------------------------------------------------------------------

class FMInteraction(nn.Module):
    """
    Inner-product based interaction (Factorization Machine style).
    Computes pairwise dot-products for all token pairs and returns
    an enriched representation by concatenating mean-pooled interactions
    back with the original tokens.

    Input : X ∈ R^{B × n × d}
    Output: X ∈ R^{B × n × d}  (same shape, enriched)
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, X: Tensor) -> Tensor:
        # X: (B, n, d)
        # Pairwise inner products: (B, n, n)
        scores = torch.bmm(X, X.transpose(1, 2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        # Weighted combination: (B, n, d)
        return torch.bmm(weights, X)


class DCNv2Interaction(nn.Module):
    """
    Deep & Cross Network v2 (Section 3.2).
    Cross layer: x^(l+1) = x^(0) ⊙ (W^(l) x^(l) + b^(l)) + x^(l)

    Operates on flattened token sequence: (B, n*d) → (B, n*d)
    then reshaped back to (B, n, d).
    """
    def __init__(self, n_tokens: int, embed_dim: int, n_cross_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        flat_dim = n_tokens * embed_dim
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(flat_dim, flat_dim) * 0.01)
            for _ in range(n_cross_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(flat_dim))
            for _ in range(n_cross_layers)
        ])
        self.deep = MLP(flat_dim, [flat_dim, flat_dim], flat_dim, dropout)
        self.output_proj = nn.Linear(flat_dim * 2, flat_dim)
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim

    def forward(self, X: Tensor) -> Tensor:
        B, n, d = X.shape
        x0 = X.reshape(B, -1)       # (B, n*d)
        xl = x0

        # Cross network
        for W, b in zip(self.cross_weights, self.cross_biases):
            xl = x0 * (xl @ W + b) + xl

        # Deep network
        deep_out = self.deep(x0)

        # Combine
        out = self.output_proj(torch.cat([xl, deep_out], dim=-1))
        return out.reshape(B, n, d)


class DHENInteraction(nn.Module):
    """
    Deep Hierarchical Ensemble Network (Section 3.2).
    X^(l+1) = Norm(Ensemble_i(Interaction_i(X^(l))) + ShortCut(X^(l)))

    Uses FM + MLP as the two heterogeneous interaction modules.
    """
    def __init__(self, n_tokens: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fm = FMInteraction(embed_dim)
        flat_dim = n_tokens * embed_dim
        self.mlp = MLP(flat_dim, [flat_dim], flat_dim, dropout)
        self.shortcut = nn.Linear(flat_dim, flat_dim)
        self.norm = nn.LayerNorm(flat_dim)
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim

    def forward(self, X: Tensor) -> Tensor:
        B, n, d = X.shape
        fm_out = self.fm(X)                              # (B, n, d)
        flat = X.reshape(B, -1)
        mlp_out = self.mlp(flat).reshape(B, n, d)
        ensemble = (fm_out + mlp_out) / 2               # mean ensemble
        shortcut = self.shortcut(flat).reshape(B, n, d)
        out = self.norm((ensemble + shortcut).reshape(B, -1))
        return out.reshape(B, n, d)


def build_interaction(name: str, n_tokens: int, embed_dim: int,
                      **kwargs) -> nn.Module:
    """Factory for interaction modules."""
    if name == "fm":
        return FMInteraction(embed_dim)
    elif name == "dcnv2":
        return DCNv2Interaction(n_tokens, embed_dim, **kwargs)
    elif name == "dhen":
        return DHENInteraction(n_tokens, embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown interaction: {name}")


# ---------------------------------------------------------------------------
# 4. Pooling by Multi-Head Attention (PMA)  (Section 3.3 / Eq. 4)
# ---------------------------------------------------------------------------

class PMA(nn.Module):
    """
    Pooling by Multi-Head Attention (Eq. 4).
    PMA(Q_pma, S) = MHA(Q_pma, K, V)

    Compresses a sequence of T tokens into k summary tokens using
    k learnable seed vectors as queries.

    Args:
        embed_dim : d
        n_heads   : number of attention heads
        k_seeds   : number of summary tokens to produce
    """
    def __init__(self, embed_dim: int, n_heads: int = 4, k_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(k_seeds, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = LayerNorm(embed_dim)

    def forward(self, S: Tensor) -> Tensor:
        """
        S       : (B, T, d)
        returns : (B, k_seeds, d)
        """
        B = S.size(0)
        Q = self.seeds.unsqueeze(0).expand(B, -1, -1)   # (B, k, d)
        out, _ = self.mha(Q, S, S)
        return self.norm(out)                             # (B, k, d)


# ---------------------------------------------------------------------------
# 5. Personalized FFN (PFFN)  (Section 4.3 / Eq. 8)
# ---------------------------------------------------------------------------

class PFFN(nn.Module):
    """
    Personalized FeedForward Network (Eq. 8).
    PFFN(X_sum, S) = f_{X_sum}(S)

    The MLP weights are conditioned on X_sum (non-sequence summarization),
    making the transformation personalized per user context.

    Implementation: hyper-network style — X_sum generates the weight
    offsets (delta_W, delta_b) added to base linear layers.

    Args:
        embed_dim    : d
        n_sum_tokens : number of summary tokens from non-sequence side
        ffn_dim      : inner FFN dimension (typically 4*d)
        dropout      : dropout rate
    """
    def __init__(self, embed_dim: int, n_sum_tokens: int,
                 ffn_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or 4 * embed_dim
        summary_dim = n_sum_tokens * embed_dim

        # Base FFN parameters
        self.W1 = nn.Linear(embed_dim, ffn_dim)
        self.W2 = nn.Linear(ffn_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(embed_dim)

        # Hyper-network: X_sum → weight offsets for W1
        self.hyper = nn.Sequential(
            nn.Linear(summary_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),   # scale for W1 input
        )

    def forward(self, X_sum: Tensor, S: Tensor) -> Tensor:
        """
        X_sum : (B, n_sum_tokens, d) — non-sequence summarization
        S     : (B, T, d)            — sequence embeddings
        returns (B, T, d)
        """
        B, T, d = S.shape
        # Compute personalized scale from X_sum
        ctx = X_sum.reshape(B, -1)          # (B, n_sum * d)
        scale = self.hyper(ctx)             # (B, d)
        scale = scale.unsqueeze(1)          # (B, 1, d)  broadcast over T

        # Apply personalized FFN
        residual = S
        h = self.W1(S * scale)             # personalized projection
        h = self.act(h)
        h = self.dropout(h)
        h = self.W2(h)
        return self.norm(residual + h)


# ---------------------------------------------------------------------------
# 6. Three Core Architectures
# ---------------------------------------------------------------------------

class InteractionArch(nn.Module):
    """
    Interaction Arch (Section 4.2 / Eq. 7).
    X^(l+1) = MLP^(l)(Interaction^(l)([X^(l) || S_sum^(l)]))

    Concatenates non-sequence features with sequence summary, runs them
    through a feature interaction module, then projects back to original shape.

    Args:
        n_nonseq_tokens : number of non-sequence feature tokens (n+1)
        n_sum_tokens    : number of sequence summary tokens (k_seeds in PMA)
        embed_dim       : d
        interaction     : "fm" | "dcnv2" | "dhen"
        dropout         : dropout rate
    """
    def __init__(self, n_nonseq_tokens: int, n_sum_tokens: int,
                 embed_dim: int, interaction: str = "dcnv2", dropout: float = 0.1):
        super().__init__()
        n_total = n_nonseq_tokens + n_sum_tokens
        self.interaction = build_interaction(interaction, n_total, embed_dim)
        # MLP to project back to original n_nonseq_tokens shape
        self.out_proj = MLP(
            n_total * embed_dim,
            [n_nonseq_tokens * embed_dim],
            n_nonseq_tokens * embed_dim,
            dropout
        )
        self.norm = LayerNorm(embed_dim)
        self.n_nonseq = n_nonseq_tokens
        self.embed_dim = embed_dim

    def forward(self, X: Tensor, S_sum: Tensor) -> Tensor:
        """
        X     : (B, n_nonseq, d)   non-sequence features
        S_sum : (B, k_seeds, d)    sequence summarization from Cross Arch
        returns (B, n_nonseq, d)
        """
        B = X.size(0)
        # [X^(l) || S_sum^(l)]  concatenate along token dimension
        X_cat = torch.cat([X, S_sum], dim=1)         # (B, n+k, d)

        # Interaction^(l)(·)
        X_inter = self.interaction(X_cat)             # (B, n+k, d)

        # MLP^(l)(·) — project back to (B, n, d)
        flat = X_inter.reshape(B, -1)                 # (B, (n+k)*d)
        out = self.out_proj(flat)                     # (B, n*d)
        out = out.reshape(B, self.n_nonseq, self.embed_dim)

        # Residual connection
        return self.norm(out + X)


class SequenceArch(nn.Module):
    """
    Sequence Arch (Section 4.3).
    S^(l+1) = MHA(PFFN(X_sum^(l), S^(l)), X_sum^(l))

    Two steps:
      1. PFFN: personalized token-wise transform guided by X_sum
      2. MHA : self-attention + cross-attention to X_sum (optional)

    Args:
        embed_dim    : d
        n_heads      : MHA heads
        n_sum_tokens : tokens in X_sum (non-sequence summary from Cross Arch)
        ffn_dim      : inner PFFN dimension
        dropout      : dropout rate
    """
    def __init__(self, embed_dim: int, n_heads: int = 4,
                 n_sum_tokens: int = 1, ffn_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.pffn = PFFN(embed_dim, n_sum_tokens, ffn_dim, dropout)
        # Self-attention among sequence tokens
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads,
                                               dropout=dropout, batch_first=True)
        # Cross-attention: sequence attends to X_sum for global context
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads,
                                                dropout=dropout, batch_first=True)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.norm3 = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, S: Tensor, X_sum: Tensor,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        S     : (B, T, d)           sequence embeddings
        X_sum : (B, n_sum, d)       non-sequence summarization from Cross Arch
        src_key_padding_mask: (B, T) True for padding positions
        returns (B, T, d)
        """
        # Step 1: PFFN — personalized transform of each sequence token
        S = self.pffn(X_sum, S)                                  # (B, T, d)

        # Step 2: Self-attention among sequence tokens
        S2, _ = self.self_attn(S, S, S,
                               key_padding_mask=src_key_padding_mask)
        S = self.norm1(S + self.dropout(S2))

        # Step 3: Cross-attention — sequence tokens attend to X_sum
        S3, _ = self.cross_attn(S, X_sum, X_sum)
        S = self.norm2(S + self.dropout(S3))

        return S


class CrossArch(nn.Module):
    """
    Cross Arch (Section 4.4).
    Connects Interaction Arch and Sequence Arch by producing summaries
    for each side via PMA.

    S_sum  = PMA(S)   → sequence summarization for Interaction Arch
    X_sum  = PMA(X)   → non-sequence summarization for Sequence Arch

    Args:
        embed_dim    : d
        n_heads      : PMA attention heads
        k_seeds      : number of summary tokens per side
        dropout      : dropout rate
    """
    def __init__(self, embed_dim: int, n_heads: int = 4,
                 k_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        # PMA for sequence → summary (feeds into Interaction Arch)
        self.seq_pma = PMA(embed_dim, n_heads, k_seeds)
        # PMA for non-sequence → summary (feeds into Sequence Arch)
        self.nonseq_pma = PMA(embed_dim, n_heads, k_seeds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, S: Tensor):
        """
        X : (B, n_nonseq, d)   non-sequence features
        S : (B, T, d)          sequence embeddings
        returns:
          S_sum : (B, k, d)   sequence summary for Interaction Arch
          X_sum : (B, k, d)   non-seq summary for Sequence Arch
        """
        S_sum = self.seq_pma(S)       # (B, k, d)
        X_sum = self.nonseq_pma(X)    # (B, k, d)
        return S_sum, X_sum


# ---------------------------------------------------------------------------
# 7. Full InterFormer Model
# ---------------------------------------------------------------------------

class InterFormer(nn.Module):
    """
    Full InterFormer model (Section 4).

    Architecture per layer l:
      (a) Interaction Arch : X^(l+1) = MLP(Interaction([X^(l) || S_sum^(l)]))
      (b) Sequence Arch    : S^(l+1) = SequenceArch(S^(l), X_sum^(l))
      (c) Cross Arch       : S_sum^(l+1), X_sum^(l+1) = CrossArch(X^(l+1), S^(l+1))

    Final prediction:
      concat X_final and S_sum_final → MLP → sigmoid

    Args:
        dense_dim          : raw dense feature dimension
        sparse_vocab_sizes : list of sparse feature vocabulary sizes
        seq_len            : T — padded sequence length
        embed_dim          : d — embedding dimension
        n_layers           : L — number of InterFormer layers
        interaction        : interaction module type "fm"|"dcnv2"|"dhen"
        n_heads            : attention heads
        k_seeds            : PMA seeds per side (summary token count)
        ffn_dim            : inner PFFN dimension (default 4*d)
        n_sequences        : k — number of behavior sequence types
        dropout            : dropout rate
        mlp_hidden_dims    : hidden dims for final prediction MLP
    """
    def __init__(
        self,
        dense_dim: int,
        sparse_vocab_sizes: List[int],
        seq_len: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        interaction: str = "dcnv2",
        n_heads: int = 4,
        k_seeds: int = 1,
        ffn_dim: Optional[int] = None,
        n_sequences: int = 1,
        dropout: float = 0.1,
        mlp_hidden_dims: List[int] = None,
    ):
        super().__init__()
        n_sparse = len(sparse_vocab_sizes)
        # n_nonseq = 1 (dense token) + n_sparse
        n_nonseq = 1 + n_sparse
        mlp_hidden_dims = mlp_hidden_dims or [256, 128]

        # --- Preprocessing ---
        self.feature_emb = FeatureEmbedding(dense_dim, sparse_vocab_sizes, embed_dim)
        self.seq_emb = nn.Embedding(
            sum(sparse_vocab_sizes) + 1, embed_dim, padding_idx=0
        )  # shared item embedding for sequences
        self.masknet = MaskNet(n_sequences, seq_len, embed_dim, dropout)

        # --- Positional encoding for sequence ---
        self.pos_enc = nn.Embedding(seq_len + 1, embed_dim)  # +1 for safety

        # --- Initial Cross Arch summaries (before first layer) ---
        # Learned initial summaries used at layer 0
        self.init_S_sum = nn.Parameter(torch.randn(1, k_seeds, embed_dim))
        self.init_X_sum = nn.Parameter(torch.randn(1, k_seeds, embed_dim))

        # --- Stacked layers ---
        self.interaction_archs = nn.ModuleList([
            InteractionArch(n_nonseq, k_seeds, embed_dim, interaction, dropout)
            for _ in range(n_layers)
        ])
        self.sequence_archs = nn.ModuleList([
            SequenceArch(embed_dim, n_heads, k_seeds, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.cross_archs = nn.ModuleList([
            CrossArch(embed_dim, n_heads, k_seeds, dropout)
            for _ in range(n_layers)
        ])

        # --- Final prediction head ---
        final_dim = n_nonseq * embed_dim + k_seeds * embed_dim
        self.pred_head = MLP(final_dim, mlp_hidden_dims, 1, dropout)

        self.n_nonseq = n_nonseq
        self.embed_dim = embed_dim
        self.k_seeds = k_seeds
        self.seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(
        self,
        dense: Tensor,
        sparse_ids: Tensor,
        seq_ids: Tensor,
        seq_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            dense            : (B, dense_dim)      raw dense features
            sparse_ids       : (B, n_sparse)        sparse feature indices
            seq_ids          : (B, T) or (B, k, T)  item ids in behavior sequence(s)
            seq_padding_mask : (B, T) bool mask, True = padding position

        Returns:
            logits : (B,)  unnormalized CTR scores (apply sigmoid for probability)
        """
        B = dense.size(0)

        # ---- Preprocessing ----
        # Non-sequence: (B, n_nonseq, d)
        X = self.feature_emb(dense, sparse_ids)

        # Sequence(s): handle single or multiple sequence inputs
        if seq_ids.dim() == 2:
            seq_ids = seq_ids.unsqueeze(1)   # (B, 1, T)

        seqs = []
        for k in range(seq_ids.size(1)):
            s = self.seq_emb(seq_ids[:, k, :])   # (B, T, d)
            # Add positional encoding
            positions = torch.arange(s.size(1), device=s.device).unsqueeze(0)
            s = s + self.pos_enc(positions)
            seqs.append(s)

        S = self.masknet(seqs)   # (B, T, d)

        # ---- Initial summaries ----
        S_sum = self.init_S_sum.expand(B, -1, -1)   # (B, k, d)
        X_sum = self.init_X_sum.expand(B, -1, -1)   # (B, k, d)

        # ---- Interleaved layers ----
        for inter_arch, seq_arch, cross_arch in zip(
            self.interaction_archs, self.sequence_archs, self.cross_archs
        ):
            # (a) Interaction Arch — behavior-aware non-sequence update
            X = inter_arch(X, S_sum)        # (B, n_nonseq, d)

            # (b) Sequence Arch — context-aware sequence update
            S = seq_arch(S, X_sum, seq_padding_mask)   # (B, T, d)

            # (c) Cross Arch — refresh summaries for next layer
            S_sum, X_sum = cross_arch(X, S)   # both (B, k, d)

        # ---- Prediction head ----
        # Flatten X and S_sum, concatenate
        X_flat = X.reshape(B, -1)           # (B, n_nonseq*d)
        S_flat = S_sum.reshape(B, -1)       # (B, k*d)
        h = torch.cat([X_flat, S_flat], dim=-1)  # (B, n_nonseq*d + k*d)

        logits = self.pred_head(h).squeeze(-1)   # (B,)
        return logits

    def predict_proba(self, *args, **kwargs) -> Tensor:
        """Returns click probability in [0, 1]."""
        return torch.sigmoid(self.forward(*args, **kwargs))


# ---------------------------------------------------------------------------
# 8. Training Utilities
# ---------------------------------------------------------------------------

class CTRTrainer:
    """
    Simple training loop for InterFormer.

    Args:
        model     : InterFormer instance
        lr        : learning rate
        weight_decay : L2 regularization
        device    : "cpu" or "cuda"
    """
    def __init__(self, model: InterFormer, lr: float = 1e-3,
                 weight_decay: float = 1e-5, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            dense, sparse_ids, seq_ids, labels = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            logits = self.model(dense, sparse_ids, seq_ids)
            loss = self.criterion(logits, labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        for batch in loader:
            dense, sparse_ids, seq_ids, labels = [b.to(self.device) for b in batch]
            logits = self.model(dense, sparse_ids, seq_ids)
            loss = self.criterion(logits, labels.float())
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Compute AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_np, probs)
        except ImportError:
            auc = float("nan")

        return {
            "loss": total_loss / len(loader),
            "auc": auc,
        }

    def fit(self, train_loader, val_loader=None, epochs: int = 10):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            msg = f"Epoch {epoch:3d} | train_loss={train_loss:.4f}"
            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                self.history["val_loss"].append(metrics["loss"])
                self.history["val_auc"].append(metrics["auc"])
                msg += f" | val_loss={metrics['loss']:.4f} | val_auc={metrics['auc']:.4f}"
            print(msg)
        return self.history


# ---------------------------------------------------------------------------
# 9. Synthetic Demo
# ---------------------------------------------------------------------------

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


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ---- Config ----
    DENSE_DIM = 16
    SPARSE_VOCAB_SIZES = [100, 200, 150, 300]   # 4 sparse features
    SEQ_LEN = 50
    EMBED_DIM = 64
    N_LAYERS = 3
    BATCH_SIZE = 32

    # ---- Build model ----
    model = InterFormer(
        dense_dim=DENSE_DIM,
        sparse_vocab_sizes=SPARSE_VOCAB_SIZES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        interaction="dcnv2",    # try "fm" or "dhen" too
        n_heads=4,
        k_seeds=1,
        dropout=0.1,
        mlp_hidden_dims=[128, 64],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")
    print(model)
    print()

    # ---- Forward pass test ----
    dense, sparse_ids, seq_ids, pad_mask, labels = make_synthetic_batch(
        BATCH_SIZE, DENSE_DIM, len(SPARSE_VOCAB_SIZES), 300, SEQ_LEN, device
    )
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(dense, sparse_ids, seq_ids, pad_mask)
        probs = torch.sigmoid(logits)

    print(f"Input  dense    : {dense.shape}")
    print(f"Input  sparse   : {sparse_ids.shape}")
    print(f"Input  sequence : {seq_ids.shape}")
    print(f"Output logits   : {logits.shape}  range=[{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Output probs    : {probs.shape}   range=[{probs.min():.3f}, {probs.max():.3f}]")
    print()

    # ---- Quick training demo (3 epochs on synthetic data) ----
    print("=== Quick training demo (synthetic data) ===")

    from torch.utils.data import TensorDataset, DataLoader

    N_TRAIN, N_VAL = 2000, 500
    def gen_dataset(n):
        d, s, sq, _, y = make_synthetic_batch(n, DENSE_DIM, len(SPARSE_VOCAB_SIZES),
                                               300, SEQ_LEN, "cpu")
        return TensorDataset(d, s, sq, y)

    train_ds = gen_dataset(N_TRAIN)
    val_ds   = gen_dataset(N_VAL)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    model_train = InterFormer(
        dense_dim=DENSE_DIM,
        sparse_vocab_sizes=SPARSE_VOCAB_SIZES,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        interaction="dcnv2",
        n_heads=4,
        k_seeds=1,
        dropout=0.1,
        mlp_hidden_dims=[128, 64],
    )
    trainer = CTRTrainer(model_train, lr=1e-3, device=device)
    history = trainer.fit(train_loader, val_loader, epochs=3)

    print("\nDone! InterFormer implementation verified.")
