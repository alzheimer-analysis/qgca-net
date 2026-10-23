"""QGCA-Net（以及仅语音的 QGCA-Speech 变体）。

流程：单模态时序编码 -> 质量感知门控（先抑制）-> 带掩码的双向跨模态注意力（后交互）-> 共享 FFN 融合
      -> 注意力池化 -> 风险头 + 数据集专属严重度头（+ 训练时的域对抗分支）。

训练期组件：门控熵正则项、整序列模态 Dropout（p_md = 0.15）、域对抗（GRL）。
"""
from typing import Dict

import torch
import torch.nn as nn

from .cross_attention import CrossModalFusion
from .encoders import EncoderBank
from .gating import QualityGate, gate_entropy_loss
from .heads import AttentivePooling, MultiTaskHeads


class QGCANet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        d = mc.d
        self.bank = EncoderBank(cfg)
        self.names = self.bank.names
        self.groups = self.bank.groups                    # 建模名 -> 物理通道 T/A/V
        self.gate = QualityGate(self.names, d, mc.gate_hidden, mc.gate_mode)
        self.fusion = CrossModalFusion(self.names, d, mc.cross_heads, mc.cross_layers, mc.attn_dropout, mc.ffn_mult,
                                       mc.directions, use_attention=(mc.fusion == "cross_attn"))
        self.pool = AttentivePooling(d, mc.pooling)
        self.heads = MultiTaskHeads(d, dropout=mc.ffn_dropout, shared_severity=bool(mc.get("shared_severity", False)))
        self.p_md = float(mc.modality_dropout)

    # --- 优化器参数分组：预训练编码层 1e-5，其余 1e-3 ---
    def pretrained_parameters(self):
        return self.bank.pretrained_parameters()

    def _modality_dropout(self, m: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """以概率 p_md 把某模态在整条序列上的掩码清零（等价于置零其特征与掩码）。若一个样本三个模态全被丢弃，
        则随机恢复其中一个，避免整条样本无输入。"""
        if not self.training or self.p_md <= 0:
            return m
        names = list(m.keys())
        B = m[names[0]].size(0)
        keep = {n: (torch.rand(B, device=m[n].device) >= self.p_md).float() for n in names}
        all_dropped = sum(keep.values()) == 0
        if all_dropped.any():
            pick = torch.randint(0, len(names), (B,), device=all_dropped.device)
            for i, n in enumerate(names):
                keep[n] = torch.where(all_dropped & (pick == i), torch.ones_like(keep[n]), keep[n])
        return {n: m[n] * keep[n][:, None] for n in names}

    def forward(self, batch: Dict, grl_coef: float = 1.0) -> Dict[str, torch.Tensor]:
        seq_mask = batch["seq_mask"]
        H = self.bank(batch)
        q = {n: batch["q"][self.groups[n]] for n in self.names}
        m = self._modality_dropout({n: batch["m"][self.groups[n]] for n in self.names})

        Hp, G = self.gate(H, q, m)                                   # h' = g · m · h
        Z = self.fusion(Hp, m)                                       # 跨模态注意力 + FFN 融合
        valid = seq_mask
        z, alpha = self.pool(Z, valid)
        logit, reg, dom = self.heads(z, batch["dataset"], grl_coef)
        out = {"logit": logit, "reg": reg, "domain_logits": dom, "z": z, "alpha": alpha,
               "gates": torch.stack([G[n] for n in self.names], -1),
               "masks": torch.stack([m[n] for n in self.names], -1)}
        out["ent_loss"] = gate_entropy_loss(G, m, seq_mask) if self.gate.mode in ("gate_q", "gate_content") \
            else logit.new_zeros(())
        return out


def build_qgca_speech_cfg(cfg):
    """QGCA-Speech：同一套门控与双向注意力，作用于 wav2vec2 与对数梅尔 CNN-GRU 两路语音流。
    两路共享语音通道的保真度指标（group = A）。"""
    from copy import deepcopy
    from ..config import Cfg
    c = deepcopy(cfg)
    c.model.modalities = Cfg({"A1": Cfg({"kind": "wav2vec", "group": "A"}),
                              "A2": Cfg({"kind": "logmel", "group": "A"})})
    return c
