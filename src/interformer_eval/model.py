"""
InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction
Paper: https://arxiv.org/abs/2411.09852

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
    Gating(X) = sigmoid(X * MLP(X))

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
    """
    def __init__(self, dense_dim: int, sparse_vocab_sizes: List[int], embed_dim: int):
        super().__init__()
        self.dense_proj = nn.Linear(dense_dim, embed_dim)
        self.sparse_embs = nn.ModuleList([
            nn.Embedding(vs, embed_dim) for vs in sparse_vocab_sizes
        ])
        self.embed_dim = embed_dim

    def forward(self, dense: Tensor, sparse_ids: Tensor) -> Tensor:
        dense_emb = self.dense_proj(dense).unsqueeze(1)
        sparse_embs = [
            self.sparse_embs[i](sparse_ids[:, i]).unsqueeze(1)
            for i in range(sparse_ids.size(1))
        ]
        return torch.cat([dense_emb] + sparse_embs, dim=1)


class MaskNet(nn.Module):
    """
    Unifies k sequences and filters noise (Eq. 6).
    MaskNet(S) = MLP_lce(S * MLP_mask(S))
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
        n_cls_tokens: int = 4,
        n_pma_tokens: int = 2,
        n_recent_tokens: int = 2,
        ffn_dim: Optional[int] = None,
        n_sequences: int = 1,
        seq_vocab_size: int = None,
        dropout: float = 0.1,
        mlp_hidden_dims: List[int] = None,
    ):
        super().__init__()
        n_sparse = len(sparse_vocab_sizes)
        n_nonseq = 1 + n_sparse
        n_seq_sum = n_cls_tokens + n_pma_tokens + n_recent_tokens
        mlp_hidden_dims = mlp_hidden_dims or [256, 128]
        max_seq_len = seq_len + n_cls_tokens

        self.n_layers = n_layers
        self.n_cls = n_cls_tokens
        self.n_nonseq = n_nonseq
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.feature_emb = FeatureEmbedding(dense_dim, sparse_vocab_sizes, embed_dim)
        if seq_vocab_size is None:
            seq_vocab_size = sum(sparse_vocab_sizes) + 1
        self.seq_emb = nn.Embedding(seq_vocab_size, embed_dim, padding_idx=0)
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
    ) -> Tensor:
        B = dense.size(0)

        X = self.feature_emb(dense, sparse_ids)

        if seq_ids.dim() == 2:
            seq_ids = seq_ids.unsqueeze(1)

        seqs = []
        for k in range(seq_ids.size(1)):
            s = self.seq_emb(seq_ids[:, k, :])
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