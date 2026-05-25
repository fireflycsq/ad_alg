"""OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer.

Paper: "OneTrans: Unified Feature Interaction and Sequence Modeling with
One Transformer in Industrial Recommender" (WWW 2026).

Train-time dims annotated with concrete demo values:
  B = batch_size (e.g. 32)
  d = d_model (e.g. 256)
  L_S = total S-tokens (e.g. 203 = 4×50 + 3[SEP])
  L_NS = num_ns_tokens (e.g. 12)
  emb_dim = 16, head_dim = d // H = 64 (for H=4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# RMSNorm (pre-norm in every OneTrans block)
# ═══════════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)              # (B, L, d)
        return x * rms * self.weight                                                # (B, L, d)


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-Split NS Tokenizer (Paper Section 3.2.1, Eq.7)
# ═══════════════════════════════════════════════════════════════════════════════


class AutoSplitNSTokenizer(nn.Module):

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        dense_dim: int,
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        embs_raw = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            embs_raw.append(None if skip else nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs_raw if e is not None])

        self._emb_index = []
        real_idx = 0
        for e in embs_raw:
            self._emb_index.append(real_idx if e is not None else -1)
            if e is not None:
                real_idx += 1

        num_features = len(feature_specs)
        total_input_dim = num_features * emb_dim + dense_dim

        self.proj = nn.Sequential(
            nn.Linear(total_input_dim, num_ns_tokens * d_model),
            nn.LayerNorm(num_ns_tokens * d_model),
        )

        self.has_dense = dense_dim > 0
        if self.has_dense:
            self.dense_proj = nn.Sequential(nn.Linear(dense_dim, dense_dim), nn.LayerNorm(dense_dim))

    def forward(self, int_feats: torch.Tensor, dense_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, int_dim) + (B, dense_dim)  →  (B, L_NS, d).

        int_dim = 103 (user_int 80 + item_int 23)
        dense_dim = 646 (user_dense)
        total_input_dim = 60(离散特征数) × emb_dim(16) + 646 = 1606
        """
        B = int_feats.shape[0]                                                         # B=32
        all_embs = []
        for i, (vs, offset, length) in enumerate(self.feature_specs):
            real_idx = self._emb_index[i]
            if real_idx == -1:
                all_embs.append(int_feats.new_zeros(B, self.emb_dim))                  # (B, 16)
            else:
                emb = self.embs[real_idx]
                if length == 1:
                    all_embs.append(emb(int_feats[:, offset].long()))                   # (B, 16)
                else:
                    vals = int_feats[:, offset:offset + length].long()                  # (B, length)
                    e = emb(vals)                                                       # (B, length, 16)
                    mask = (vals != 0).float().unsqueeze(-1)                           # (B, length, 1)
                    all_embs.append((e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1))  # (B, 16)

        if self.has_dense and dense_feats is not None:
            all_embs.append(self.dense_proj(dense_feats))                               # (B, 646) → (B, 646)

        cat = torch.cat(all_embs, dim=-1)                                               # (B, 1606)
        return self.proj(cat).view(B, self.num_ns_tokens, -1)                          # (B, 12, 256)


# ═══════════════════════════════════════════════════════════════════════════════
# Sequential Tokenizer (Paper Section 3.2.2)
# ═══════════════════════════════════════════════════════════════════════════════


class SeqTokenizer(nn.Module):

    def __init__(
        self,
        seq_vocab_sizes: Dict[str, List[int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.domains = sorted(seq_vocab_sizes.keys())
        self._embs = nn.ModuleDict()
        self._emb_index: Dict[str, List[int]] = {}
        self._proj: nn.ModuleDict = nn.ModuleDict()

        for domain in self.domains:
            vs_list = seq_vocab_sizes[domain]
            embs_raw = []
            for vs in vs_list:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                embs_raw.append(None if skip else nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            self._embs[domain] = nn.ModuleList([e for e in embs_raw if e is not None])

            idx_map = []
            real_idx = 0
            for e in embs_raw:
                idx_map.append(real_idx if e is not None else -1)
                if e is not None:
                    real_idx += 1
            self._emb_index[domain] = idx_map

            self._proj[domain] = nn.Sequential(
                nn.Linear(len(vs_list) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        self.sep_token = nn.Parameter(torch.zeros(1, d_model))

    def forward(
        self, seq_data: Dict[str, torch.Tensor], seq_lens: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-domain embed & project → merge with [SEP] tokens.

        Input:
          seq_data[domain]:  (B, n_feats, L)   domain_a: (32, 9,  50)
          seq_lens[domain]:  (B,)               original lengths per sample

        Returns:
          s_tokens: (B, total_L, d)     = (32, 203, 256)
          s_mask:   (B, total_L)        True = padding position
        """
        B = seq_data[self.domains[0]].shape[0]                                         # B=32
        device = seq_data[self.domains[0]].device
        all_tokens, all_masks = [], []

        for idx, domain in enumerate(self.domains):                # domain_a, domain_b, domain_c, domain_d
            seq = seq_data[domain]                                                      # (B, n_feats, L)  e.g. (32, 9, 50)
            lengths = seq_lens[domain]                                                  # (B,)
            n_feats, L = seq.shape[1], seq.shape[2]                                    # n_feats=9, L=50

            # Embed each sideinfo feature → concat → per-domain MLP
            emb_list = []
            for sid in range(n_feats):                                                  # for each sideinfo feature
                real_idx = self._emb_index[domain][sid] if sid < len(self._emb_index[domain]) else -1
                if real_idx == -1:
                    emb_list.append(seq.new_zeros(B, L, self._proj[domain][0].in_features // n_feats))  # (B, 50, 16)
                else:
                    emb_list.append(self._embs[domain][real_idx](seq[:, sid, :]))      # (B, 50, 16)

            cat_emb = torch.cat(emb_list, dim=-1)                                       # (B, 50, n_feats×16) = (B, 50, 144)
            tokens = self._proj[domain](cat_emb)                                                # (B, 50, 256)  — per-event linear projection

            # Padding mask: True where position >= actual sequence length
            mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)  # (B, 50)

            all_tokens.append(tokens)                                                    # (B, 50, 256)
            all_masks.append(mask)                                                       # (B, 50)

            # [SEP] token between domains (not after the last)
            if idx < len(self.domains) - 1:
                all_tokens.append(self.sep_token.expand(B, 1, -1))                      # (B, 1, 256)
                all_masks.append(torch.zeros(B, 1, dtype=torch.bool, device=device))    # (B, 1) all valid

        # Concat: domain_a(50) + [SEP](1) + domain_b(50) + [SEP](1) + domain_c(50) + [SEP](1) + domain_d(50) = 203
        s_tokens = torch.cat(all_tokens, dim=1)                                          # (B, 203, 256)
        s_mask   = torch.cat(all_masks, dim=1)                                           # (B, 203)
        return s_tokens, s_mask


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed Causal Attention (Paper Section 3.3.1, Eq.11-12)
# ═══════════════════════════════════════════════════════════════════════════════


class MixedCausalAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, L_NS: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.L_NS = L_NS
        self.dropout = dropout
        assert d_model % num_heads == 0

        # Shared projections for S-tokens
        self.W_q_s = nn.Linear(d_model, d_model)
        self.W_k_s = nn.Linear(d_model, d_model)
        self.W_v_s = nn.Linear(d_model, d_model)
        self.W_o_s = nn.Linear(d_model, d_model)

        # Token-specific projections for each NS-token (L_NS=12 sets)
        self.W_q_ns = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(L_NS)])
        self.W_k_ns = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(L_NS)])
        self.W_v_ns = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(L_NS)])
        self.W_o_ns = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(L_NS)])

    def _proj_qkv(self, x: torch.Tensor, L_S: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, K, V for ALL tokens using mixed parameterization.

        x: (B, L_total, d)   — [S₁...S_LS | NS₁...NS_LNS]

        S-tokens [0:L_S) use shared W_q_s/W_k_s/W_v_s
        NS-token j at position L_S+j uses its own W_q_ns[j]/W_k_ns[j]/W_v_ns[j]
        """
        B, L_total, D = x.shape                                                         # B=32, L_total=215, D=256
        L_NS = self.L_NS                                                                # 12

        # K, V for ALL tokens (always full sequence)
        K_parts, V_parts = [], []
        if L_S > 0:
            K_parts.append(self.W_k_s(x[:, :L_S, :]))                                    # (B, L_S, 256)
            V_parts.append(self.W_v_s(x[:, :L_S, :]))                                    # (B, L_S, 256)
        for j in range(L_NS):
            K_parts.append(self.W_k_ns[j](x[:, L_S + j:L_S + j + 1, :]))                # (B, 1, 256) each
            V_parts.append(self.W_v_ns[j](x[:, L_S + j:L_S + j + 1, :]))                # (B, 1, 256) each
        K = torch.cat(K_parts, dim=1)                                                    # (B, L_total, 256)
        V = torch.cat(V_parts, dim=1)                                                    # (B, L_total, 256)

        # Q for ALL tokens (will be filtered later if pyramid mode)
        Q_parts = []
        if L_S > 0:
            Q_parts.append(self.W_q_s(x[:, :L_S, :]))                                    # (B, L_S, 256)
        for j in range(L_NS):
            Q_parts.append(self.W_q_ns[j](x[:, L_S + j:L_S + j + 1, :]))                # (B, 1, 256) each
        Q = torch.cat(Q_parts, dim=1)                                                    # (B, L_total, 256)

        return Q, K, V

    def _proj_output(self, x: torch.Tensor, L_S_current: int) -> torch.Tensor:
        """Apply mixed output projection W_o.

        x: (B, L_out, d)  — output after attention
        L_S_current: # of S-tokens at this layer

        First L_S_current positions use shared W_o_s.
        Remaining L_NS positions each use token-specific W_o_ns[j].
        """
        B, L_out, D = x.shape                                                            # B=32, L_out varies
        L_NS = self.L_NS

        O_parts = []
        s_count = min(L_S_current, L_out)                                                # # of S-tokens in output
        if s_count > 0:
            O_parts.append(self.W_o_s(x[:, :s_count, :]))                                # (B, s_count, 256) shared
        ns_start = s_count
        for j in range(L_NS):
            if ns_start + j < L_out:
                O_parts.append(self.W_o_ns[j](x[:, ns_start + j:ns_start + j + 1, :]))  # (B, 1, 256) per-token
        return torch.cat(O_parts, dim=1)                                                 # (B, L_out, 256)

    def _reshape_mha(self, t: torch.Tensor) -> torch.Tensor:
        """(B, L, d) → (B, H, L, head_dim)"""
        B, L, _ = t.shape
        return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mixed causal attention forward.

        x:                  (B, L_total, d)    S-tokens first, then NS-tokens
        key_padding_mask:   (B, L_total)       True = padding (do not attend)
        query_mask:         (B, L_total)       True = this position issues a query (pyramid)

        Returns:
            (B, Lq, d)    Lq = query_mask.sum()/B  or  L_total (if no pyramid)
        """
        B, L_total, D = x.shape                                                         # B=32, L_total=215, D=256
        L_NS = self.L_NS                                                                # 12
        L_S = L_total - L_NS                                                            # 203

        # Step 1: Compute Q, K, V for all tokens with mixed parameterization
        Q_full, K_full, V_full = self._proj_qkv(x, L_S)                                 # 3 × (B, L_total, 256)

        # Step 2: Select queries (pyramid pruning)
        if query_mask is not None:
            Lq = int(query_mask.sum().item() // B)                                      # e.g. 207 for layer 0 pyramid
            Q = Q_full[query_mask].view(B, Lq, D)                                        # (B, Lq, 256)
        else:
            Lq = L_total                                                                 # no pruning
            Q = Q_full                                                                   # (B, L_total, 256)

        # Step 3: Reshape to (B, H, L, head_dim)
        Q = self._reshape_mha(Q)                                                         # (B, 4, Lq, 64)
        K = self._reshape_mha(K_full)                                                    # (B, 4, L_total, 64)
        V = self._reshape_mha(V_full)                                                    # (B, 4, L_total, 64)

        # Step 4: Causal mask — position i attends to positions ≤ i
        if query_mask is not None:
            # Pyramid: each query position attends to all K positions ≤ itself
            q_positions = query_mask.nonzero(as_tuple=False)[:, 1].view(B, Lq)          # (B, Lq)  — absolute positions
            k_positions = torch.arange(L_total, device=x.device).unsqueeze(0).unsqueeze(0)  # (1, 1, L_total)
            causal = k_positions <= q_positions.unsqueeze(-1)                           # (B, Lq, L_total)
            causal = causal.unsqueeze(1)                                                  # (B, 1, Lq, L_total)
        else:
            causal = torch.tril(torch.ones(L_total, L_total, dtype=torch.bool, device=x.device))  # (L_total, L_total)
            causal = causal.unsqueeze(0).unsqueeze(0)                                    # (1, 1, L_total, L_total)

        # Step 5: Combine causal + padding mask → SDPA format (B, H, Lq, L_total)
        if key_padding_mask is not None:
            pad_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)                      # (B, 1, 1, L_total)  True=attend
            attn_mask = causal & pad_mask                                                # (B, 1, Lq, L_total)
        else:
            attn_mask = causal
        attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)                        # (B, 4, Lq, L_total)

        # Step 6: Flash Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, dropout_p=dropout_p)  # (B, 4, Lq, 64)
        out = torch.nan_to_num(out, nan=0.0)

        # Step 7: Back to (B, Lq, d) → mixed output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)                           # (B, Lq, 256)
        L_S_out = Lq - L_NS                                                              # remaining S-tokens after pyramid
        return self._proj_output(out, L_S_out)                                          # (B, Lq, 256)


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed FFN (Paper Section 3.3.2, Eq.13)
# ═══════════════════════════════════════════════════════════════════════════════


class MixedFFN(nn.Module):

    def __init__(self, d_model: int, L_NS: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.L_NS = L_NS
        hidden_dim = d_model * hidden_mult                                                # 1024

        # Shared FFN for all S-tokens
        self.ffn_s = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout),
        )
        # Token-specific FFN for each NS-token
        self.ffn_ns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model), nn.Dropout(dropout),
            )
            for _ in range(L_NS)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixed FFN: S-tokens share ffn_s, each NS-token has its own ffn_ns[j].

        x: (B, L_total, d)   where L_total = L_S_current + L_NS
        Returns: (B, L_total, d)
        """
        L_NS = self.L_NS
        L_S = x.shape[1] - L_NS                                                         # current S-token count at this layer

        out_parts = []
        if L_S > 0:
            out_parts.append(self.ffn_s(x[:, :L_S, :]))                                  # (B, L_S, 256)  shared W¹, W²
        for j in range(L_NS):
            out_parts.append(self.ffn_ns[j](x[:, L_S + j:L_S + j + 1, :]))              # (B, 1, 256)   per-token W¹ⱼ, W²ⱼ
        return torch.cat(out_parts, dim=1)                                               # (B, L_total, 256)


# ═══════════════════════════════════════════════════════════════════════════════
# OneTrans Block (Paper Section 3.3, Fig.2b)
# ═══════════════════════════════════════════════════════════════════════════════


class OneTransBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, L_NS: int,
                 hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)                                                     # Pre-norm for attention
        self.attn  = MixedCausalAttention(d_model, num_heads, L_NS, dropout)
        self.norm2 = RMSNorm(d_model)                                                     # Pre-norm for FFN
        self.ffn   = MixedFFN(d_model, L_NS, hidden_mult, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """OneTransBlock = Pre-norm RMSNorm → MixedCausalAttention → MixedFFN.

        Paper Eq.4-5:
          Z^(n)   = MixedMHA (Norm(X^(n-1))) + X^(n-1)
          X^(n)   = MixedFFN (Norm(Z^(n)))   + Z^(n)

        x:                  (B, L_in,  d)   incoming tokens
        key_padding_mask:   (B, L_in)       True = padding
        query_mask:         (B, L_in)       True = issue query (pyramid pruning)

        Returns:
          x_out:             (B, L_out, d)  where L_out = query_mask.sum() or L_in
          new_mask:          (B, L_out)     downsampled padding mask
        """
        # ── Attention sub-block (Pre-Norm + residual) ──
        # Eq.4: Z = MixedMHA(Norm(X)) + X
        attn_out = self.attn(self.norm1(x), key_padding_mask, query_mask)                # (B, L_out, 256)

        if query_mask is not None:
            residual = x[query_mask].view(attn_out.shape)                                 # (B, L_out, 256)  — same positions as queries
        else:
            residual = x                                                                   # (B, L_in, 256)
        x_out = residual + attn_out                                                        # (B, L_out, 256)

        # ── FFN sub-block (Pre-Norm + residual) ──
        # Eq.5: X' = MixedFFN(Norm(Z)) + Z
        x_out = x_out + self.ffn(self.norm2(x_out))                                       # (B, L_out, 256)

        # Downsample padding mask to match output
        if key_padding_mask is not None and query_mask is not None:
            new_mask = key_padding_mask[query_mask].view(x_out.shape[0], x_out.shape[1]) # (B, L_out)
        else:
            new_mask = key_padding_mask                                                    # (B, L_in)  unchanged

        return x_out, new_mask


# ═══════════════════════════════════════════════════════════════════════════════
# Pyramid Schedule (Paper Section 3.4)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_pyramid_targets(L_S_initial: int, L_NS: int, num_layers: int) -> List[int]:
    """Target S-query count per layer: linear from L_S → L_NS, rounded to 32.

    Example for L_S=203, L_NS=12, N=6:
      Layer 0: 203 → Layer 1: 165 → Layer 2: 127
      Layer 3: 89  → Layer 4: 51  → Layer 5: 12
    """
    targets = []
    for n in range(num_layers):
        if num_layers == 1:
            k = L_NS
        else:
            k = int(L_S_initial - n * (L_S_initial - L_NS) / (num_layers - 1))
            k = max(round(k / 32) * 32, L_NS)
        targets.append(max(k, L_NS))
    return targets


def build_query_mask(L_S_current: int, L_NS: int, target_s_queries: int,
                     B: int, device: torch.device) -> torch.Tensor:
    """Create boolean mask selecting the last k S-tokens + ALL NS-tokens.

    Returns: (B, L_total)  where L_total = L_S_current + L_NS
    """
    L_total = L_S_current + L_NS
    num_s_q = min(target_s_queries, L_S_current)                                         # actual # of S-tokens to keep
    s_start = L_S_current - num_s_q                                                      # tail of S

    mask = torch.zeros(B, L_total, dtype=torch.bool, device=device)                      # (B, L_total)   all False
    mask[:, s_start:L_S_current] = True                                                   # last k S-tokens
    mask[:, L_S_current:L_total] = True                                                    # all NS-tokens
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# OneTrans Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class OneTrans(nn.Module):

    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        d_model: int = 256,
        emb_dim: int = 16,
        num_layers: int = 6,
        num_heads: int = 4,
        num_ns_tokens: int = 12,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_ns_tokens = num_ns_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_domains = sorted(seq_vocab_sizes.keys())

        # ── NS Tokenizer: Auto-Split ──
        all_int_specs = list(user_int_feature_specs)
        item_offset_start = sum(length for _, _, length in user_int_feature_specs)
        for vs, offset, length in item_int_feature_specs:
            all_int_specs.append((vs, item_offset_start + offset, length))

        self.ns_tokenizer = AutoSplitNSTokenizer(
            feature_specs=all_int_specs,
            dense_dim=user_dense_dim + item_dense_dim,
            emb_dim=emb_dim,
            d_model=d_model,
            num_ns_tokens=num_ns_tokens,
            emb_skip_threshold=emb_skip_threshold,
        )

        self._user_int_dim   = sum(length for _, _, length in user_int_feature_specs)     # 80
        self._item_int_dim   = sum(length for _, _, length in item_int_feature_specs)     # 23
        self._user_dense_dim = user_dense_dim                                              # 646
        self._item_dense_dim = item_dense_dim                                              # 0

        # ── S-Tokenizer ──
        if self.seq_domains:
            self.seq_tokenizer = SeqTokenizer(seq_vocab_sizes, emb_dim, d_model, emb_skip_threshold)
        else:
            self.seq_tokenizer = None

        # ── OneTrans Blocks (lazy build when L_S is known) ──
        self.blocks: nn.ModuleList = nn.ModuleList()
        self._pyramid_targets: List[int] = []
        self._L_S_initial: int = 0
        self._blocks_built: bool = False
        self._hidden_mult = hidden_mult
        self._dropout = dropout

        # ── Task Tower ──
        self.output_proj = nn.Sequential(
            nn.Linear(num_ns_tokens * d_model, d_model),                                  # (L_NS*d) → d   e.g. 3072 → 256
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),                                                         # 256 → 1
        )

        self._init_params()

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx] = 0

    def _build_blocks(self, L_S: int, device: torch.device) -> None:
        if self._blocks_built and self._L_S_initial == L_S:
            return
        self._L_S_initial = L_S
        L_NS = self.num_ns_tokens

        self.blocks = nn.ModuleList([
            OneTransBlock(self.d_model, self.num_heads, L_NS, self._hidden_mult, self._dropout)
            for _ in range(self.num_layers)
        ]).to(device)
        self._pyramid_targets = compute_pyramid_targets(L_S, L_NS, self.num_layers)
        self._blocks_built = True

    def get_sparse_params(self) -> List[nn.Parameter]:
        ptrs = {m.weight.data_ptr() for m in self.modules() if isinstance(m, nn.Embedding)}
        return [p for p in self.parameters() if p.data_ptr() in ptrs]

    def get_dense_params(self) -> List[nn.Parameter]:
        ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in ptrs]

    def forward(
        self,
        user_int_feats:   torch.Tensor,        # (B, 80)
        item_int_feats:   torch.Tensor,        # (B, 23)
        user_dense_feats: torch.Tensor,        # (B, 646)
        item_dense_feats: torch.Tensor,        # (B, 0)
        seq_data: Optional[Dict[str, torch.Tensor]] = None,   # {domain: (B, n_feats, L)}
        seq_lens: Optional[Dict[str, torch.Tensor]] = None,   # {domain: (B,)}
    ) -> torch.Tensor:
        """OneTrans full forward pass. Returns (B, 1) logits.

        Shape flow (demo OneTransS: L=6, d=256, H=4, L_NS=12):
        ┌────────────────────────────────────────────────────────────────┐
        │ Step 1: Tokenization                                           │
        │   int(103) + dense(646) → AutoSplitNS → (B, 12, 256)          │
        │   seq(4 domains × 50) + 3[SEP] → SeqTokenizer → (B, 203, 256) │
        ├────────────────────────────────────────────────────────────────┤
        │ Step 2: Concatenate                                            │
        │   [S-tokens | NS-tokens] = (B, 215, 256)                       │
        ├────────────────────────────────────────────────────────────────┤
        │ Step 3: Pyramid of OneTransBlocks × 6                          │
        │   Layer 0: (B, 215, 256) → (B, 207, 256)   S-q: 203→195     │
        │   Layer 1: (B, 207, 256) → (B, 177, 256)   S-q: 195→165     │
        │   Layer 2: (B, 177, 256) → (B, 139, 256)   S-q: 165→127     │
        │   Layer 3: (B, 139, 256) → (B, 101, 256)   S-q: 127→89      │
        │   Layer 4: (B, 101, 256) → (B,  63, 256)   S-q: 89→51       │
        │   Layer 5: (B,  63, 256) → (B,  24, 256)   S-q: 51→12       │
        ├────────────────────────────────────────────────────────────────┤
        │ Step 4: Task Tower                                             │
        │   Last L_NS tokens → flatten → (B, 12×256) → MLP → (B, 1)    │
        └────────────────────────────────────────────────────────────────┘
        """
        B = user_int_feats.shape[0]                                                     # B=32
        device = user_int_feats.device

        # ── Step 1: NS tokenization ──
        all_int   = torch.cat([user_int_feats, item_int_feats], dim=1)                   # (B, 103)
        all_dense = torch.cat([user_dense_feats, item_dense_feats], dim=1)               # (B, 646)
        ns_tokens = self.ns_tokenizer(all_int, all_dense)                                # (B, 12, 256)

        # ── Step 2: S-tokenization ──
        if self.seq_tokenizer is not None and seq_data:
            s_tokens, s_mask = self.seq_tokenizer(seq_data, seq_lens)                    # (B, 203, 256), (B, 203)
        else:
            s_tokens = torch.empty(B, 0, self.d_model, device=device)                    # (B, 0, 256)
            s_mask   = torch.empty(B, 0, dtype=torch.bool, device=device)                 # (B, 0)

        L_S = s_tokens.shape[1]                                                          # 203 (or 0 if no sequences)
        L_NS = self.num_ns_tokens                                                         # 12
        self._build_blocks(L_S, device)

        # ── Step 3: [S-tokens | NS-tokens] ──
        if L_S > 0:
            x = torch.cat([s_tokens, ns_tokens], dim=1)                                  # (B, 215, 256)
            key_padding_mask = torch.cat([
                s_mask,
                torch.zeros(B, L_NS, dtype=torch.bool, device=device)                    # (B, 12)  NS always valid
            ], dim=1)                                                                      # (B, 215)
        else:
            x = ns_tokens                                                                  # (B, 12, 256)
            key_padding_mask = torch.zeros(B, L_NS, dtype=torch.bool, device=device)      # (B, 12)

        # ── Step 4: Pyramid of OneTrans blocks ──
        for layer_idx, block in enumerate(self.blocks):
            L_total = x.shape[1]                                                          # e.g. 215, 207, 177, ...
            L_S_current = L_total - L_NS                                                  # e.g. 203, 195, 165, ...
            target_k = self._pyramid_targets[layer_idx]                                   # e.g. 195, 165, 127, ...

            if L_S_current > 0 and target_k < L_S_current:
                q_mask = build_query_mask(L_S_current, L_NS, target_k, B, device)         # (B, L_total)  select last k S + all NS
            else:
                q_mask = None                                                               # no pruning needed

            x, key_padding_mask = block(x, key_padding_mask, q_mask)                      # (B, L_out, 256), (B, L_out)

        # ── Step 5: Task Tower → logit ──
        if x.shape[1] >= L_NS:
            ns_out = x[:, -L_NS:, :]                                                       # (B, 12, 256)  take last L_NS tokens
        else:
            ns_out = x                                                                      # fallback
        pooled = ns_out.reshape(B, -1)                                                     # (B, 12×256) = (B, 3072)
        return self.output_proj(pooled)                                                    # (B, 1)

    def predict(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference: returns (logits, embedding_placeholder)."""
        logits = self.forward(*args, **kwargs)
        return logits, logits