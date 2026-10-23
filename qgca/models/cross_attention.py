"""带掩码的双向跨模态注意力（论文 Eq. CA 与随后的聚合/融合公式）。

  CA_{S→M} = softmax( Q K^T / sqrt(d_k) + log M_S ) V_S,   Q = H'_M W_Q,  K = H'_S W_K,  V_S = H'_S W_V
  Ĥ_M = LN( H'_M + Σ_{S≠M} Dropout(CA_{S→M}) )
  Z = LN( U + FFN(U) ),  U = [Ĥ_T; Ĥ_A; Ĥ_V] W_F

M_S 把模态 S 的可用性掩码广播到键维度，被掩码的键注意力权重为 0（log 0 用 -1e4 近似）。
8 个头，2 层堆叠；双向共 6 个方向。
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


class DirectionalCrossAttention(nn.Module):
    """S → M：M 作查询，S 作键/值。"""

    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.wq, self.wk, self.wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.wo = nn.Linear(d, d)

    def forward(self, h_m: torch.Tensor, h_s: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        B, n, d = h_m.shape
        q = self.wq(h_m).view(B, n, self.h, self.dk).transpose(1, 2)
        k = self.wk(h_s).view(B, n, self.h, self.dk).transpose(1, 2)
        v = self.wv(h_s).view(B, n, self.h, self.dk).transpose(1, 2)
        logm = (1.0 - key_mask)[:, None, None, :] * NEG                        # log M_S
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=logm.to(q.dtype))
        o = self.wo(o.transpose(1, 2).reshape(B, n, d))
        has_key = (key_mask.sum(1) > 0).to(o.dtype)[:, None, None]             # 整条序列都被掩码 => 无信息
        return o * has_key


def make_directions(names: List[str], mode: str) -> List[Tuple[str, str]]:
    """返回 (source, target) 列表。bidirectional：所有有序对；unidirectional：环形 3 个方向。"""
    if mode == "bidirectional":
        return [(s, t) for t in names for s in names if s != t]
    # 单向：每个模态恰被一个源查询（A→T, V→A, T→V 的环形排列，顺序取 names 的轮转）
    return [(names[(i + 1) % len(names)], names[i]) for i in range(len(names))]


class CrossModalLayer(nn.Module):
    def __init__(self, names: List[str], d: int, heads: int, dropout: float, directions: List[Tuple[str, str]]):
        super().__init__()
        self.names, self.directions = names, directions
        self.ca = nn.ModuleDict({f"{s}2{t}": DirectionalCrossAttention(d, heads) for s, t in directions})
        self.ln = nn.ModuleDict({n: nn.LayerNorm(d) for n in names})
        self.drop = nn.Dropout(dropout)

    def forward(self, H: Dict[str, torch.Tensor], M: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for t in self.names:
            acc = H[t]
            add = 0.0
            for s, tt in self.directions:
                if tt == t:
                    add = add + self.drop(self.ca[f"{s}2{t}"](H[t], H[s], M[s]))
            out[t] = self.ln[t](acc + add)
        return out


class CrossModalFusion(nn.Module):
    """多层跨模态注意力 + 共享前馈融合。fusion='concat' 时去掉注意力，仅拼接融合（消融）。"""

    def __init__(self, names: List[str], d: int, heads: int, layers: int, dropout: float, ffn_mult: int,
                 direction_mode: str = "bidirectional", use_attention: bool = True):
        super().__init__()
        self.names, self.use_attention = names, use_attention
        if use_attention:
            dirs = make_directions(names, direction_mode)
            self.layers = nn.ModuleList([CrossModalLayer(names, d, heads, dropout, dirs) for _ in range(layers)])
        self.wf = nn.Linear(len(names) * d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, Hp: Dict[str, torch.Tensor], M: Dict[str, torch.Tensor]) -> torch.Tensor:
        H = Hp
        if self.use_attention:
            for layer in self.layers:
                H = layer(H, M)
        u = self.wf(torch.cat([H[n] for n in self.names], -1))
        return self.ln(u + self.ffn(u))
