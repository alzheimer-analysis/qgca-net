"""质量感知门控（论文 Eq. gate）与门控熵正则项。

  g_{M,i} = σ( W_{M,2} ReLU( W_{M,1}[h_{M,i}; q_{M,i}] + b_{M,1} ) + b_{M,2} )      (每个模态独立参数，隐层 128)
  h'_{M,i} = g_{M,i} · m_{M,i} · h_{M,i}

gate_mode:
  gate_q       完整模型：门控输入 [h; q]
  gate_content 内容门控：只输入 h（无 q）
  none         无门控：h' = m · h
  q_feature    无门控，q 作为输入特征：h' = m · W[h; q]
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class QualityGate(nn.Module):
    def __init__(self, names: List[str], d: int, hidden: int, mode: str):
        super().__init__()
        self.mode = mode
        self.names = names
        if mode in ("gate_q", "gate_content"):
            din = d + (1 if mode == "gate_q" else 0)
            self.net = nn.ModuleDict({n: nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                                      for n in names})
        elif mode == "q_feature":
            self.qproj = nn.ModuleDict({n: nn.Linear(d + 1, d) for n in names})

    def forward(self, H: Dict[str, torch.Tensor], q: Dict[str, torch.Tensor], m: Dict[str, torch.Tensor]
                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """H/q/m 按调用方给出的建模名索引；返回 (H', g)。g 在无门控时为 m 本身。"""
        Hp, G = {}, {}
        for n in self.names:
            h, qq, mm = H[n], q[n], m[n]
            if self.mode == "gate_q":
                g = torch.sigmoid(self.net[n](torch.cat([h, qq.unsqueeze(-1).to(h.dtype)], -1))).squeeze(-1)
            elif self.mode == "gate_content":
                g = torch.sigmoid(self.net[n](h)).squeeze(-1)
            elif self.mode == "q_feature":
                h = self.qproj[n](torch.cat([h, qq.unsqueeze(-1).to(h.dtype)], -1))
                g = torch.ones_like(mm)
            else:
                g = torch.ones_like(mm)
            Hp[n] = (g * mm).unsqueeze(-1) * h
            G[n] = g
        return Hp, G


def gate_entropy_loss(G: Dict[str, torch.Tensor], M: Dict[str, torch.Tensor], seq_mask: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """L_ent = (1/n) Σ_i Σ_M p_{M,i} log p_{M,i}，p_i = softmax(g_{T,i}, g_{A,i}, g_{V,i}) 取自可用模态（负熵）。

    对每个样本在其有效话语上平均，再对批取平均。至少有一个可用模态的话语才计入。
    """
    names = list(G.keys())
    g = torch.stack([G[n] for n in names], -1).float()               # [B,n,K]
    avail = torch.stack([M[n] for n in names], -1) > 0                # [B,n,K]
    logits = g.masked_fill(~avail, -1e4)
    p = torch.softmax(logits, -1) * avail
    neg_ent = (p * torch.log(p + eps)).sum(-1)                        # [B,n]
    valid = (avail.any(-1) & (seq_mask > 0)).float()
    per_sample = (neg_ent * valid).sum(1) / valid.sum(1).clamp(min=1.0)
    return per_sample.mean()
