import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class KVCachedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_cache_len=2048, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** 0.5   # fixed

    def forward(self, query, key, value, cache=None, use_causal_mask=True):
        B, Tq, _ = query.size()

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # ---- Cache correctly ----
        if cache and cache["key"] is not None:
            K = torch.cat([cache["key"], K], dim=1)
            V = torch.cat([cache["value"], V], dim=1)

        cache_len = K.shape[1] - Tq

        # ---- Split heads properly ----
        Q = self.split(Q)
        K = self.split(K)
        V = self.split(V)

        # ---- Compute scores correctly ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # ---- Causal mask ----
        if use_causal_mask:
            total = cache_len + Tq
            mask = torch.tril(torch.ones(Tq, total, device=Q.device)).bool()
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = scores.softmax(dim=-1)

        if self.training:   # dropout only during training
            attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = self.merge(out)
        out = self.out_proj(out)

        # ---- Cache stores FULL history ----
        new_cache = {
            "key":  K.reshape(B, -1, self.d_model),
            "value":V.reshape(B, -1, self.d_model)

        }

        return out, new_cache

    def split(self, x):
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1,2)

    def merge(self, x):
        B, H, T, D = x.size()
        return x.transpose(1,2).reshape(B,T,H*D)
