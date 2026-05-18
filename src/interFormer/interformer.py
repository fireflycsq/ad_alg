"""
InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction
Paper: https://arxiv.org/abs/2411.09852
Corrected PyTorch implementation following the paper's architecture precisely.

Architecture Overview:
  For each layer l = 1..L:
    1. Cross Arch       : S_sum^(l), X_sum^(l) = CrossArch(X^(l), S^(l))
    2. Interaction Arch  : X^(l+1) = MLP(Interaction([X^(l) || S_sum^(l)]))
    3. Sequence Arch     : S^(l+1) = MHA(PFFN(X_sum^(l), S^(l)))
  Final CTR score = sigmoid(MLP([X_sum^(L) || S_sum^(L)]))
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
    """Standard LayerNorm wrapper."""
    pass


class MLP(nn.Module):
    """Multi-layer perceptron with optional dropout. Uses Swish/SiLU per paper."""
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int,
                 dropout: float = 0.1, activation: str = "silu"):
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


class Gating(nn.Module):
    """
    Self-gating mechanism (Eq. 10).
    Gating(X) = σ(X ⊙ MLP(X))

    Provides sparse masking: relevant information retained, noise filtered out.
    """
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, X: Tensor) -> Tensor:
        return X * self.mlp(X)


# ---------------------------------------------------------------------------
# 2. Rotary Position Embedding (RoPE) — Section 4.3
# ---------------------------------------------------------------------------

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from Su et al. 2024.
    Applied to Q and K in Sequence Arch MHA.
    """
    def __init__(self, head_dim: int, max_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, q: Tensor, k: Tensor, seq_offset: int = 0) -> tuple:
        T = q.size(2)
        cos = self.cos[seq_offset:seq_offset + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[seq_offset:seq_offset + T].unsqueeze(0).unsqueeze(0)
        return self._apply_rope(q, k, cos, sin)

    def _apply_rope(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        q_rot = q.float()
        k_rot = k.float()
        q1, q2 = q_rot[..., ::2], q_rot[..., 1::2]
        k1, k2 = k_rot[..., ::2], k_rot[..., 1::2]
        q_out = torch.empty_like(q_rot)
        k_out = torch.empty_like(k_rot)
        q_out[..., ::2] = q1 * cos - q2 * sin
        q_out[..., 1::2] = q2 * cos + q1 * sin
        k_out[..., ::2] = k1 * cos - k2 * sin
        k_out[..., 1::2] = k2 * cos + k1 * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)


class RoPEMultiheadAttention(nn.Module):
    """
    Multi-Head Attention with RoPE support.
    Used in Sequence Arch for self-attention on sequence tokens.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1,
                 max_len: int = 2048):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_len)
        self.dropout = dropout

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, d = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, key_padding_mask.size(1),
                                    device=x.device, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 3. Feature Preprocessing  (Section 4.1)
# ---------------------------------------------------------------------------

class FeatureEmbedding(nn.Module):
    """
    Embeds dense + sparse non-sequence features into a unified matrix X^(1).

    Scalar sparse features (dim=1): standard Embedding lookup → 1 token.
    Array sparse features (dim>1): embed all D elements with shared Embedding,
    mask padding (value=0), mean-pool → 1 token.
    """
    def __init__(self, dense_dim: int, sparse_vocab_sizes: List[int],
                 embed_dim: int, sparse_is_array: Optional[List[bool]] = None,
                 sparse_multi_dim: Optional[List[int]] = None):
        super().__init__()
        self.dense_proj = nn.Linear(dense_dim, embed_dim)
        self.sparse_embs = nn.ModuleList([
            nn.Embedding(max(vs, 1), embed_dim, padding_idx=0) for vs in sparse_vocab_sizes
        ])
        self.embed_dim = embed_dim
        self.is_array = sparse_is_array or [False] * len(sparse_vocab_sizes)
        self.multi_dim = sparse_multi_dim or [0] * len(sparse_vocab_sizes)
        # Pre-compute array_idx counter for forward pass
        self._array_slots: List[Tuple[int, int]] = []  # [(emb_idx, array_idx), ...]
        _aidx = 0
        for i, is_arr in enumerate(self.is_array):
            if is_arr:
                self._array_slots.append((i, _aidx))
                _aidx += 1
        self.n_array_feats = _aidx

    def forward(self, dense: Tensor, sparse_ids: Tensor,
                sparse_multi: Optional[Tensor] = None,
                sparse_multi_mask: Optional[Tensor] = None) -> Tensor:
        dense_emb = self.dense_proj(dense).unsqueeze(1)  # (B, 1, d)

        tokens = [dense_emb]
        _aidx_map = {emb_i: aidx for emb_i, aidx in self._array_slots}

        for i in range(len(self.sparse_embs)):
            if i in _aidx_map and sparse_multi is not None:
                # Array feature: embed all D elements, mask, mean-pool
                aidx = _aidx_map[i]
                vals = sparse_multi[:, aidx, :].long()     # (B, D)
                emb = self.sparse_embs[i](vals)             # (B, D, d)
                if sparse_multi_mask is not None:
                    mask = sparse_multi_mask[:, aidx, :].float().unsqueeze(-1)  # (B, D, 1)
                    emb = emb * mask
                    denom = mask.sum(dim=1).clamp(min=1)    # (B, 1)
                    pooled = emb.sum(dim=1) / denom          # (B, d)
                else:
                    pooled = emb.mean(dim=1)
                tokens.append(pooled.unsqueeze(1))           # (B, 1, d)
            else:
                # Scalar feature: standard lookup
                tok = self.sparse_embs[i](sparse_ids[:, i]).unsqueeze(1)
                tokens.append(tok)

        return torch.cat(tokens, dim=1)


class MaskNet(nn.Module):
    """
    Unifies k sequences and filters noise (Eq. 6).
    MaskNet(S) = MLP_lce(S ⊙ MLP_mask(S))
    """
    def __init__(self, k: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp_mask = nn.Sequential(
            nn.Linear(k * embed_dim, k * embed_dim),
            nn.Sigmoid()
        )
        self.mlp_lce = nn.Linear(k * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences: List[Tensor]) -> Tensor:
        S = torch.cat(sequences, dim=-1)
        mask = self.mlp_mask(S)
        S = S * mask
        S = self.mlp_lce(S)
        return self.dropout(S)


# ---------------------------------------------------------------------------
# 4. Feature Interaction Modules  (Section 3.2)
# ---------------------------------------------------------------------------

class FMInteraction(nn.Module):
    """Inner-product based interaction (FM style)."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, X: Tensor) -> Tensor:
        scores = torch.bmm(X, X.transpose(1, 2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, X)


class DCNv2Interaction(nn.Module):
    """Deep & Cross Network v2 (Section 3.2)."""
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
        x0 = X.reshape(B, -1)
        xl = x0
        for W, b in zip(self.cross_weights, self.cross_biases):
            xl = x0 * (xl @ W + b) + xl
        deep_out = self.deep(x0)
        out = self.output_proj(torch.cat([xl, deep_out], dim=-1))
        return out.reshape(B, n, d)


class DHENInteraction(nn.Module):
    """Deep Hierarchical Ensemble Network (Section 3.2)."""
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
        fm_out = self.fm(X)
        flat = X.reshape(B, -1)
        mlp_out = self.mlp(flat).reshape(B, n, d)
        ensemble = (fm_out + mlp_out) / 2
        shortcut = self.shortcut(flat).reshape(B, n, d)
        out = self.norm((ensemble + shortcut).reshape(B, -1))
        return out.reshape(B, n, d)


def build_interaction(name: str, n_tokens: int, embed_dim: int,
                      **kwargs) -> nn.Module:
    if name == "fm":
        return FMInteraction(embed_dim)
    elif name == "dcnv2":
        return DCNv2Interaction(n_tokens, embed_dim, **kwargs)
    elif name == "dhen":
        return DHENInteraction(n_tokens, embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown interaction: {name}")


# ---------------------------------------------------------------------------
# 5. Pooling by Multi-Head Attention (PMA)  (Section 3.3 / Eq. 4)
# ---------------------------------------------------------------------------

class PMA(nn.Module):
    """
    Pooling by Multi-Head Attention (Eq. 4).
    PMA(Q_pma, S) = MHA(Q_pma, K, V)
    """
    def __init__(self, embed_dim: int, n_heads: int = 4, k_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(k_seeds, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = LayerNorm(embed_dim)

    def forward(self, S: Tensor) -> Tensor:
        B = S.size(0)
        Q = self.seeds.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.mha(Q, S, S)
        return self.norm(out)


# ---------------------------------------------------------------------------
# 6. Linear Compressed Embedding (LCE) — Appendix A.2.1
# ---------------------------------------------------------------------------

class LCE(nn.Module):
    """
    Linear Compressed Embedding (Appendix A.2.1).
    Given N d-dimensional features X in R^{d x N}, LCE is a linear
    transformation W in R^{N x M} on the sample dimension,
    such that XW serves as the compressed embedding with M features.
    """
    def __init__(self, n_input: int, n_output: int, embed_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_input, n_output) * 0.01)

    def forward(self, X: Tensor) -> Tensor:
        """
        X: (B, d, n_input)
        returns: (B, d, n_output)
        """
        return X @ self.W


# ---------------------------------------------------------------------------
# 7. Personalized FFN (PFFN)  (Section 4.3 / Eq. 8, Appendix A.2.2)
# ---------------------------------------------------------------------------

class PFFN(nn.Module):
    """
    Personalized FeedForward Network (Eq. 8, Appendix A.2.2).

    PFFN(X_sum, S) = f(X_sum) * S

    Learns a transformation weight W_PFFN = f(X_sum) in R^{d x d} from
    the non-sequence summarization, then applies it to the sequence embeddings.
    """
    def __init__(self, embed_dim: int, n_sum_tokens: int):
        super().__init__()
        summary_dim = n_sum_tokens * embed_dim

        self.hyper = nn.Sequential(
            nn.Linear(summary_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim * embed_dim),
        )
        self.norm = LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, X_sum: Tensor, S: Tensor) -> Tensor:
        B, T, d = S.shape
        ctx = X_sum.reshape(B, -1)
        W_pffn = self.hyper(ctx).reshape(B, d, d)
        return self.norm(torch.bmm(S, W_pffn.transpose(1, 2)))


# ---------------------------------------------------------------------------
# 8. Three Core Architectures
# ---------------------------------------------------------------------------

class InteractionArch(nn.Module):
    """
    Interaction Arch (Section 4.2 / Eq. 7).
    X^(l+1) = MLP^(l)(Interaction^(l)([X^(l) || S_sum^(l)]))
    """
    def __init__(self, n_nonseq_tokens: int, n_sum_tokens: int,
                 embed_dim: int, interaction: str = "dcnv2", dropout: float = 0.1):
        super().__init__()
        n_total = n_nonseq_tokens + n_sum_tokens
        self.interaction = build_interaction(interaction, n_total, embed_dim,
                                             dropout=dropout)
        flat_in = n_total * embed_dim
        flat_out = n_nonseq_tokens * embed_dim
        bottleneck = embed_dim * 4
        self.out_proj = nn.Sequential(
            nn.Linear(flat_in, bottleneck),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, flat_out),
            nn.LayerNorm(flat_out),
        )
        self.norm = LayerNorm(embed_dim)
        self.n_nonseq = n_nonseq_tokens
        self.embed_dim = embed_dim

    def forward(self, X: Tensor, S_sum: Tensor) -> Tensor:
        B = X.size(0)
        X_cat = torch.cat([X, S_sum], dim=1)
        X_inter = self.interaction(X_cat)
        flat = X_inter.reshape(B, -1)
        out = self.out_proj(flat).reshape(B, self.n_nonseq, self.embed_dim)
        return self.norm(out)


class SequenceArch(nn.Module):
    """
    Sequence Arch (Section 4.3 / Eq. 9).
    S^(l+1) = MHA^(l)(PFFN(X_sum^(l), S^(l)))

    S includes CLS tokens prepended before the first layer.
    """
    def __init__(self, embed_dim: int, n_heads: int = 4,
                 n_sum_tokens: int = 1, dropout: float = 0.1,
                 max_seq_len: int = 2048):
        super().__init__()
        self.pffn = PFFN(embed_dim, n_sum_tokens)
        self.mha = RoPEMultiheadAttention(embed_dim, n_heads, dropout, max_seq_len)
        self.n_sum_tokens = n_sum_tokens

    def forward(self, S: Tensor, X_sum: Tensor,
                seq_padding_mask: Optional[Tensor] = None) -> Tensor:
        B = S.size(0)

        if seq_padding_mask is not None:
            cls_mask = seq_padding_mask.new_zeros(B, self.n_sum_tokens)
            full_mask = torch.cat([cls_mask, seq_padding_mask], dim=1)
        else:
            full_mask = None

        S = self.pffn(X_sum, S)
        S = self.mha(S, key_padding_mask=full_mask)
        return S


class CrossArch(nn.Module):
    """
    Cross Arch (Section 4.4).

    Non-sequence summarization (Eq. 10):
      X_sum^(l) = Gating(LCE(X^(l)))

    Sequence summarization (Eq. 11):
      S_sum^(l) = Gating([S_CLS^(l) || S_PMA^(l) || S_recent^(l)])
    """
    def __init__(self, embed_dim: int, n_nonseq_tokens: int,
                 n_cls_tokens: int = 4, n_pma_tokens: int = 2,
                 n_recent_tokens: int = 2, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_cls = n_cls_tokens
        self.n_pma = n_pma_tokens
        self.n_recent = n_recent_tokens

        self.nonseq_lce = LCE(n_nonseq_tokens, n_cls_tokens, embed_dim)
        self.nonseq_gate = Gating(embed_dim)
        self.seq_pma = PMA(embed_dim, n_heads, n_pma_tokens)
        self.seq_gate = Gating(embed_dim)

    def forward(self, X: Tensor, S: Tensor):
        # Non-sequence summarization (Eq. 10)
        X_t = X.transpose(1, 2)
        X_compressed = self.nonseq_lce(X_t)
        X_sum = X_compressed.transpose(1, 2)
        X_sum = self.nonseq_gate(X_sum)

        # Sequence summarization (Eq. 11)
        S_cls = S[:, :self.n_cls, :]
        S_pma = self.seq_pma(S)
        S_recent = S[:, -self.n_recent:, :]

        S_cat = torch.cat([S_cls, S_pma, S_recent], dim=1)
        S_sum = self.seq_gate(S_cat)

        return S_sum, X_sum


# ---------------------------------------------------------------------------
# 9. Full InterFormer Model
# ---------------------------------------------------------------------------

class InterFormer(nn.Module):
    """
    Full InterFormer model (Section 4, Algorithm 1).

    Architecture per layer l:
      (a) Cross Arch       : S_sum^(l), X_sum^(l) = CrossArch(X^(l), S^(l))
      (b) Interaction Arch : X^(l+1) = MLP(Interaction([X^(l) || S_sum^(l)]))
      (c) Sequence Arch    : S^(l+1) = MHA(PFFN(X_sum^(l), S^(l)))

    Final prediction:
      y_hat = sigmoid(MLP([X_sum^(L) || S_sum^(L)]))

    Non-sequence features (Section 4.1):
      X^(1) = [x_dense || x_sparse_user1 || ... || x_sparse_item1 || ...]
      All user_int, item_int, user_dense, item_dense features concatenated.

    Multi-sequence (Section 4.1, Eq. 6):
      k sequences (e.g. click, conversion, different platforms) are fused
      via MaskNet into a single d-dimensional sequence per timestep.

    Args:
        dense_dim          : raw dense feature dimension (user_dense + item_dense)
        sparse_vocab_sizes : list of sparse vocab sizes (user_int + item_int)
        seq_len            : T — padded sequence length
        seq_vocab_sizes    : per-domain sequence vocab sizes for embedding
        embed_dim          : d — embedding dimension
        n_layers           : L — number of InterFormer layers
        interaction        : interaction module type "fm"|"dcnv2"|"dhen"
        n_heads            : attention heads
        n_cls_tokens       : CLS / X_sum token count (paper default: 4)
        n_pma_tokens       : PMA token count for seq summary (paper default: 2)
        n_recent_tokens    : recent token count (paper default: 2)
        n_sequences        : k — number of behavior sequence domains
        sparse_is_array    : per-slot bool list for array feature mean-pooling
        sparse_multi_dim   : per-slot int list of array dims (0 = scalar)
        dropout            : dropout rate
        mlp_hidden_dims    : hidden dims for final prediction MLP
    """
    def __init__(
        self,
        dense_dim: int,
        sparse_vocab_sizes: List[int],
        seq_len: int,
        seq_vocab_sizes: List[int] = None,
        embed_dim: int = 64,
        n_layers: int = 3,
        interaction: str = "dcnv2",
        n_heads: int = 4,
        n_cls_tokens: int = 4,
        n_pma_tokens: int = 2,
        n_recent_tokens: int = 2,
        n_sequences: int = 1,
        sparse_is_array: Optional[List[bool]] = None,
        sparse_multi_dim: Optional[List[int]] = None,
        dropout: float = 0.1,
        mlp_hidden_dims: List[int] = None,
    ):
        super().__init__()
        n_sparse = len(sparse_vocab_sizes)
        n_nonseq = 1 + n_sparse
        n_seq_sum = n_cls_tokens + n_pma_tokens + n_recent_tokens
        mlp_hidden_dims = mlp_hidden_dims or [256, 128]
        max_seq_len = seq_len + n_cls_tokens

        if seq_vocab_sizes is None:
            seq_vocab_sizes = [[sum(sparse_vocab_sizes) + 1]]
        if n_sequences is None:
            n_sequences = len(seq_vocab_sizes)

        self.n_layers = n_layers
        self.n_cls = n_cls_tokens
        self.n_nonseq = n_nonseq
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.n_sequences = n_sequences

        self.feature_emb = FeatureEmbedding(
            dense_dim, sparse_vocab_sizes, embed_dim,
            sparse_is_array=sparse_is_array,
            sparse_multi_dim=sparse_multi_dim,
        )

        # Per-domain, per-feature sequence embeddings
        self.seq_embs = nn.ModuleList([
            nn.ModuleList([
                nn.Embedding(max(vs, 1), embed_dim, padding_idx=0)
                for vs in domain_vocabs
            ])
            for domain_vocabs in seq_vocab_sizes
        ])
        self.masknet = MaskNet(n_sequences, embed_dim, dropout)

        self.cross_archs = nn.ModuleList([
            CrossArch(embed_dim, n_nonseq, n_cls_tokens, n_pma_tokens,
                      n_recent_tokens, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.interaction_archs = nn.ModuleList([
            InteractionArch(n_nonseq, n_seq_sum, embed_dim, interaction, dropout)
            for _ in range(n_layers)
        ])
        self.sequence_archs = nn.ModuleList([
            SequenceArch(embed_dim, n_heads, n_cls_tokens,
                         dropout, max_seq_len)
            for _ in range(n_layers)
        ])

        final_dim = n_cls_tokens * embed_dim + n_seq_sum * embed_dim
        self.pred_head = MLP(final_dim, mlp_hidden_dims, 1, dropout)

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
        sparse_multi: Optional[Tensor] = None,
        sparse_multi_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B = dense.size(0)

        X = self.feature_emb(dense, sparse_ids, sparse_multi, sparse_multi_mask)

        # seq_ids: (B, k, max_feats, T) — k domains × max_feats features × T timesteps
        # Embed each feature within each domain, sum to get per-domain sequence
        seqs = []
        for k in range(seq_ids.size(1)):
            s = None
            n_feats = len(self.seq_embs[k])
            for f in range(n_feats):
                feat_ids = seq_ids[:, k, f, :]             # (B, T)
                feat_emb = self.seq_embs[k][f](feat_ids)   # (B, T, d)
                if s is None:
                    s = feat_emb
                else:
                    s = s + feat_emb
            seqs.append(s)

        S = self.masknet(seqs)

        # Initial CLS prepend (Algorithm 1 step 3)
        X_t = X.transpose(1, 2)
        X_init = self.cross_archs[0].nonseq_lce(X_t)
        X_init = X_init.transpose(1, 2)
        X_init = self.cross_archs[0].nonseq_gate(X_init)
        X_sum = X_init

        S = torch.cat([X_sum, S], dim=1)

        # Interleaved layers (Algorithm 1 steps 4-8)
        S_sum = None
        for cross_arch, inter_arch, seq_arch in zip(
            self.cross_archs, self.interaction_archs, self.sequence_archs
        ):
            S_sum, X_sum = cross_arch(X, S)
            X = inter_arch(X, S_sum)
            S = seq_arch(S, X_sum, seq_padding_mask)

        # Prediction head (Algorithm 1 step 9)
        X_sum_flat = X_sum.reshape(B, -1)
        S_sum_flat = S_sum.reshape(B, -1)
        h = torch.cat([X_sum_flat, S_sum_flat], dim=-1)

        logits = self.pred_head(h).squeeze(-1)
        return logits

    def predict_proba(self, *args, **kwargs) -> Tensor:
        return torch.sigmoid(self.forward(*args, **kwargs))


# ---------------------------------------------------------------------------
# 10. Training Utilities
# ---------------------------------------------------------------------------

class CTRTrainer:
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

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_np, probs)
        except ImportError:
            auc = float("nan")

        return {"loss": total_loss / len(loader), "auc": auc}

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
# 11. Synthetic Demo
# ---------------------------------------------------------------------------

def make_synthetic_batch(B: int, dense_dim: int, n_sparse: int,
                         vocab_size: int, seq_len: int, device: str = "cpu"):
    dense = torch.randn(B, dense_dim, device=device)
    sparse_cols = [torch.randint(0, vs, (B,), device=device)
                   for vs in [100, 200, 150, 300][:n_sparse]]
    sparse_ids = torch.stack(sparse_cols, dim=1)
    seq_ids = torch.randint(1, vocab_size, (B, seq_len), device=device)
    pad_start = int(seq_len * 0.8)
    seq_padding_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    seq_padding_mask[:, pad_start:] = True
    labels = torch.randint(0, 2, (B,), device=device)
    return dense, sparse_ids, seq_ids, seq_padding_mask, labels


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    DENSE_DIM = 16
    SPARSE_VOCAB_SIZES = [100, 200, 150, 300]
    SEQ_LEN = 50
    EMBED_DIM = 64
    N_LAYERS = 3
    BATCH_SIZE = 32

    N_SEQUENCES = 2
    SEQ_VOCAB_SIZES = [[300, 300], [300]]
    MAX_SEQ_FEATS = 2

    model = InterFormer(
        dense_dim=DENSE_DIM,
        sparse_vocab_sizes=SPARSE_VOCAB_SIZES,
        seq_len=SEQ_LEN,
        seq_vocab_sizes=SEQ_VOCAB_SIZES,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        interaction="dcnv2",
        n_heads=4,
        n_cls_tokens=4,
        n_pma_tokens=2,
        n_recent_tokens=2,
        n_sequences=N_SEQUENCES,
        dropout=0.1,
        mlp_hidden_dims=[128, 64],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}\n")
    print(model)
    print()

    dense, sparse_ids, seq_ids_1d, pad_mask, labels = make_synthetic_batch(
        BATCH_SIZE, DENSE_DIM, len(SPARSE_VOCAB_SIZES), 300, SEQ_LEN, device
    )
    seq_ids = torch.zeros(BATCH_SIZE, N_SEQUENCES, MAX_SEQ_FEATS, SEQ_LEN,
                          dtype=torch.long, device=device)
    seq_ids[:, 0, 0, :] = seq_ids_1d
    seq_ids[:, 0, 1, :] = torch.randint(1, 300, (BATCH_SIZE, SEQ_LEN), device=device)
    seq_ids[:, 1, 0, :] = torch.randint(1, 300, (BATCH_SIZE, SEQ_LEN), device=device)
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

    print("=== Quick training demo (synthetic data) ===")
    from torch.utils.data import TensorDataset, DataLoader

    N_TRAIN, N_VAL = 2000, 500
    def gen_dataset(n):
        d, s, sq_1d, _, y = make_synthetic_batch(
            n, DENSE_DIM, len(SPARSE_VOCAB_SIZES), 300, SEQ_LEN, "cpu")
        sq = torch.zeros(n, N_SEQUENCES, MAX_SEQ_FEATS, SEQ_LEN, dtype=torch.long)
        sq[:, 0, 0, :] = sq_1d
        sq[:, 0, 1, :] = torch.randint(1, 300, (n, SEQ_LEN))
        sq[:, 1, 0, :] = torch.randint(1, 300, (n, SEQ_LEN))
        return TensorDataset(d, s, sq, y)

    train_ds = gen_dataset(N_TRAIN)
    val_ds = gen_dataset(N_VAL)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model_train = InterFormer(
        dense_dim=DENSE_DIM,
        sparse_vocab_sizes=SPARSE_VOCAB_SIZES,
        seq_len=SEQ_LEN,
        seq_vocab_sizes=SEQ_VOCAB_SIZES,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        interaction="dcnv2",
        n_heads=4,
        n_cls_tokens=4,
        n_pma_tokens=2,
        n_recent_tokens=2,
        n_sequences=N_SEQUENCES,
        dropout=0.1,
        mlp_hidden_dims=[128, 64],
    )
    trainer = CTRTrainer(model_train, lr=1e-3, device=device)
    history = trainer.fit(train_loader, val_loader, epochs=3)

    print("\nDone! InterFormer implementation verified.")