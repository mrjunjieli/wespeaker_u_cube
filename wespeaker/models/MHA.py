# Copyright (c) 2026 Junjie LI (mrjunjieli@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F


class MultiViewSelfAttention(nn.Module):
    """
    Multi-View Self-Attention (paper: Multi-view self-attention based transformer for speaker recognition).
    - embed_dim: model embedding dim
    - num_heads: number of heads (paper experiments use 8)
    - dropout: attention dropout
    - max_window_cap: optional cap on window size (to avoid extremely large windows)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_window_cap=300):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_window_cap = max_window_cap

        # projections for Q,K,V and output
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)


    def _head_windows(self, device):
        """
        compute window sizes per head according to paper:
            w_i = 1 if i==0 else 2**i + 1
        return list of half-window sizes (radius), i.e., number of positions on each side.
        """
        half_windows = []
        for i in range(0,self.num_heads):
            w = 2 ** (i+1) + 1
            half = (w - 1) // 2
            half_windows.append(half)
        return torch.tensor(half_windows, device=device)  # shape (H,)

    def forward(self, x, mask=None):
        """
        x: (B, T, E)
        mask: optional bool mask (B, T) where True = valid token (compatible with padding)
        returns: (B, T, E)
        """
        B, T, E = x.shape
        device = x.device

        # qkv: (B, T, 3*E) -> split -> (B, T, H, head_dim)
        qkv = self.qkv_proj(x)  # (B, T, 3E)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, T, H, D)

        # transpose to (B, H, T, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # compute raw attention logits: (B, H, T, T)
        # matmul: q @ k^T
        # for memory, compute batch matmul with reshape
        # logits = torch.einsum('bhid,bhjd->bhij', q, k)  # alternative
        q_ = q * self.scale
        logits = torch.matmul(q_, k.transpose(-2, -1))  # (B, H, T, T)

        # build head-wise band mask (allowed region = True). shape (H, T, T)
        half_windows = self._head_windows(device)  # (H,)
        # broadcast index matrices (T,T)
        idx = torch.arange(T, device=device)
        # dist matrix (T, T) = abs(i - j)
        dist = (idx[None, :] - idx[:, None]).abs()  # (T, T)
        # Make mask per head by comparing dist with half_windows
        # half_windows[:, None, None] -> (H,1,1) broadcast to (H,T,T)
        allowed = dist[None, :, :].unsqueeze(0)  # temp shape (1, T, T) -> we'll compare per head
        # but simpler: construct per-head mask with broadcasting:
        # allowed_heads: (H, T, T) boolean
        half_windows_expand = half_windows[:, None, None]  # (H,1,1)
        allowed_heads = (dist[None, :, :] <= half_windows_expand)  # (H, T, T)

        # now we have allowed_heads (H, T, T), expand to (B, H, T, T) for masking logits
        allowed_heads = allowed_heads.unsqueeze(0).expand(B, -1, -1, -1)

        # apply mask: set logits outside allowed region to a large negative
        neg_inf = -1e9
        
        logits = torch.where(allowed_heads, logits, torch.tensor(neg_inf, device=device, dtype=logits.dtype))

        # if padding mask is provided, also block attention to padding positions
        if mask is not None:
            # mask: (B, T) bool where True = valid; we want to disallow attending to invalid (False)
            # create key_mask (B, 1, 1, T) to broadcast
            key_mask = (~mask).unsqueeze(1).unsqueeze(2)  # True where invalid
            logits = logits.masked_fill(key_mask, neg_inf)

        # softmax
        attn = torch.softmax(logits, dim=-1)  # (B, H, T, T)
        attn = self.attn_dropout(attn)

        # attention output
        out = torch.matmul(attn, v)  # (B, H, T, D)
        # combine heads
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, T, H, D)
        out = out.view(B, T, E)  # (B, T, E)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out


class MultiViewTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden=2048, dropout=0.2, max_window_cap=300):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiViewSelfAttention(embed_dim, num_heads, dropout=dropout, max_window_cap=max_window_cap)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (B, T, E)
        # mask: (B, T) True=valid, False=padded
        # ---- Self Attention + Residual ----
        attn_out = self.self_attn(x, mask=mask)
        x = self.norm1(x + attn_out)

        # ---- Feed-forward + Residual ----
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x