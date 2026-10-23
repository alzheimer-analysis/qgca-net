"""对比方法（论文 Compared Methods）：早期融合、晚期融合，以及重实现的 DepAudioNet、DMPF（仅语音）和
MLAN、MulT、TAMFN、TTFNet（三模态）。

说明：这些方法在原论文中的完整细节无法全部复现。三模态方法共享与 QGCA-Net 相同的 BERT / wav2vec 2.0 /
OpenFace+ViT 特征与 EncoderBank，因此差异只来自融合设计；各方法的融合模块按其核心思想实现，
结构超参数（层数、头数、隐层宽度）放在 cfg.baseline，并通过 configs/search_space.yaml 中的同一 24 配置预算调优。
基线均不使用质量指标（cfg.model.gate_mode = none，仅保留"技术性缺失"掩码）。
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import DirectionalCrossAttention
from .encoders import EncoderBank, run_rnn
from .heads import AttentivePooling, MultiTaskHeads
from .transformer import TimeGapTransformer


class BaselineBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.model.d
        self.bank = EncoderBank(cfg)
        self.names = self.bank.names
        self.groups = self.bank.groups
        self.pool = AttentivePooling(self.d, "attentive")
        self.heads = MultiTaskHeads(self.d, dropout=cfg.model.ffn_dropout,
                                    shared_severity=bool(cfg.model.get("shared_severity", False)))

    def pretrained_parameters(self):
        return self.bank.pretrained_parameters()

    def masks(self, batch) -> Dict[str, torch.Tensor]:
        return {n: batch["m"][self.groups[n]] for n in self.names}

    def finish(self, z, batch, grl_coef=1.0, **extra):
        logit, reg, dom = self.heads(z, batch["dataset"], grl_coef)
        out = {"logit": logit, "reg": reg, "domain_logits": dom, "z": z, "ent_loss": logit.new_zeros(())}
        out.update(extra)
        return out


# --------------------------------------------------------------------------- 早期融合
class EarlyFusion(BaselineBase):
    """特征级拼接：各模态话语向量逐话语拼接 -> 投影 -> Bi-GRU -> 注意力池化。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.proj = nn.Sequential(nn.Linear(len(self.names) * self.d, self.d), nn.ReLU(), nn.Dropout(cfg.model.ffn_dropout))
        self.gru = nn.GRU(self.d, self.d // 2, batch_first=True, bidirectional=True)

    def forward(self, batch, grl_coef=1.0):
        H, M = self.bank(batch), self.masks(batch)
        x = torch.cat([H[n] * M[n].unsqueeze(-1) for n in self.names], -1)
        h = run_rnn(self.gru, self.proj(x), batch["seq_mask"])
        z, _ = self.pool(h, batch["seq_mask"])
        return self.finish(z, batch, grl_coef)


# --------------------------------------------------------------------------- 晚期融合
class LateFusion(BaselineBase):
    """决策级融合：每个模态独立的池化 + 头；风险 logit 按验证表现加权投票（权重 ∝ 单模态验证宏 F1）。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.pools = nn.ModuleDict({n: AttentivePooling(self.d) for n in self.names})
        self.branch = nn.ModuleDict({n: MultiTaskHeads(self.d, dropout=cfg.model.ffn_dropout) for n in self.names})
        w = cfg.model.get("late_weights", None) or [1.0] * len(self.names)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32) / sum(w))
        self.last_branch_logits = None

    def set_branch_weights(self, f1: List[float]):
        w = torch.tensor(f1, dtype=torch.float32).clamp(min=1e-3)
        self.w.copy_(w / w.sum())

    def forward(self, batch, grl_coef=1.0):
        H, M = self.bank(batch), self.masks(batch)
        logits, regs, zs = [], [], []
        for n in self.names:
            z, _ = self.pools[n](H[n] * M[n].unsqueeze(-1), batch["seq_mask"])
            l, r, _ = self.branch[n](z, batch["dataset"], grl_coef)
            logits.append(l)
            regs.append(r)
            zs.append(z)
        L, R = torch.stack(logits, -1), torch.stack(regs, -1)
        w = self.w.to(L.dtype)
        # 缺失的模态不参与投票：权重按该样本至少含一个可用话语的模态重新归一
        avail = torch.stack([(M[n] * batch["seq_mask"]).sum(1) > 0 for n in self.names], -1).to(L.dtype)
        ww = w[None] * avail
        ww = ww / ww.sum(-1, keepdim=True).clamp(min=1e-6)
        out = self.finish(torch.stack(zs, 0).mean(0), batch, grl_coef)
        out["logit"] = (L * ww).sum(-1)
        out["reg"] = (R * ww).sum(-1)
        self.last_branch_logits = L.detach()
        out["branch_logits"] = L.detach()
        return out


# --------------------------------------------------------------------------- MLAN
class MLAN(BaselineBase):
    """多层级注意力：各模态 Bi-LSTM -> 模态级注意力（每个话语在三模态上加权）-> 时间级注意力 -> 头。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        b = cfg.baseline
        self.lstm = nn.ModuleDict({n: nn.LSTM(self.d, b.hidden // 2, num_layers=b.lstm_layers, batch_first=True,
                                              bidirectional=True) for n in self.names})
        self.mod_att = nn.Sequential(nn.Linear(b.hidden, b.hidden), nn.Tanh(), nn.Linear(b.hidden, 1, bias=False))
        self.to_d = nn.Linear(b.hidden, self.d)

    def forward(self, batch, grl_coef=1.0):
        H, M = self.bank(batch), self.masks(batch)
        S = torch.stack([run_rnn(self.lstm[n], H[n], batch["seq_mask"]) for n in self.names], 2)   # [B,n,K,h]
        mm = torch.stack([M[n] for n in self.names], 2)                                              # [B,n,K]
        e = self.mod_att(S).squeeze(-1).masked_fill(mm <= 0, -1e4)
        a = torch.softmax(e, 2) * (mm.sum(2, keepdim=True) > 0)
        fused = self.to_d((a.unsqueeze(-1) * S).sum(2))
        z, _ = self.pool(fused, batch["seq_mask"])
        return self.finish(z, batch, grl_coef)


# --------------------------------------------------------------------------- MulT
class _CrossBlock(nn.Module):
    def __init__(self, d, heads, dropout, ffn_mult=4):
        super().__init__()
        self.ca = DirectionalCrossAttention(d, heads)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src, src_mask):
        x = self.ln1(x + self.drop(self.ca(x, src, src_mask)))
        return self.ln2(x + self.drop(self.ffn(x)))


def sinusoid(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class MulT(BaselineBase):
    """多模态 Transformer：每个有序模态对一个跨模态 Transformer（目标作查询、源作键值），
    同一目标的各源输出拼接后经自注意力 Transformer，再拼接各目标并池化。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        b = cfg.baseline
        K = len(self.names)
        self.cross = nn.ModuleDict({f"{s}2{t}": nn.ModuleList([_CrossBlock(self.d, b.heads, cfg.model.ffn_dropout)
                                                               for _ in range(b.layers)])
                                    for t in self.names for s in self.names if s != t})
        self.self_tf = nn.ModuleDict({t: TimeGapTransformer(self.d * (K - 1), b.heads if (self.d * (K - 1)) % b.heads == 0 else 1,
                                                            b.layers, 4, cfg.model.ffn_dropout, time_aware=False)
                                      for t in self.names})
        self.out = nn.Linear(K * self.d * (K - 1), self.d)

    def forward(self, batch, grl_coef=1.0):
        H, M = self.bank(batch), self.masks(batch)
        n = batch["seq_mask"].size(1)
        pe = sinusoid(n, self.d, batch["seq_mask"].device).unsqueeze(0)
        H = {k: (v + pe) * M[k].unsqueeze(-1) for k, v in H.items()}
        outs = []
        for t in self.names:
            parts = []
            for s in self.names:
                if s == t:
                    continue
                x = H[t]
                for blk in self.cross[f"{s}2{t}"]:
                    x = blk(x, H[s], M[s])
                parts.append(x)
            outs.append(self.self_tf[t](torch.cat(parts, -1), batch["time"], batch["seq_mask"]))
        z, _ = self.pool(self.out(torch.cat(outs, -1)), batch["seq_mask"])
        return self.finish(z, batch, grl_coef)


# --------------------------------------------------------------------------- TAMFN
class TAMFN(BaselineBase):
    """时间感知注意力多模态融合：模态间特征提取（每个模态以其余两个模态的拼接序列为键值）
    + 时间感知融合（带话语时间间隔偏置的 Transformer）。文本流作为第三路输入。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        b = cfg.baseline
        K = len(self.names)
        self.inter = nn.ModuleDict({t: nn.ModuleList([DirectionalCrossAttention(self.d, b.heads) for _ in range(b.layers)])
                                    for t in self.names})
        self.ln = nn.ModuleDict({t: nn.LayerNorm(self.d) for t in self.names})
        self.fuse_in = nn.Linear(K * self.d, self.d)
        self.time_tf = TimeGapTransformer(self.d, b.heads, b.layers, 4, cfg.model.ffn_dropout, time_aware=True)

    def forward(self, batch, grl_coef=1.0):
        H, M = self.bank(batch), self.masks(batch)
        ext = {}
        for t in self.names:
            others = [s for s in self.names if s != t]
            kv = torch.cat([H[s] for s in others], 1)                           # [B,(K-1)n,d]
            km = torch.cat([M[s] for s in others], 1)
            x = H[t]
            for ca in self.inter[t]:
                x = self.ln[t](x + ca(x, kv, km))
            ext[t] = x
        fused = self.fuse_in(torch.cat([ext[t] for t in self.names], -1))
        h = self.time_tf(fused, batch["time"], batch["seq_mask"])
        z, _ = self.pool(h, batch["seq_mask"])
        return self.finish(z, batch, grl_coef)


# --------------------------------------------------------------------------- TTFNet
class TTFNet(BaselineBase):
    """时间感知三模态融合：每个模态先经时间间隔感知的 Transformer，再把三模态 token 加模态嵌入拼成一条序列，
    由联合时间感知 Transformer 融合并池化（被掩码的 token 不参与注意力与池化）。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        b = cfg.baseline
        self.uni = nn.ModuleDict({n: TimeGapTransformer(self.d, b.heads, 1, 4, cfg.model.ffn_dropout) for n in self.names})
        self.mod_emb = nn.Parameter(torch.zeros(len(self.names), self.d))
        nn.init.normal_(self.mod_emb, std=0.02)
        self.joint = TimeGapTransformer(self.d, b.heads, b.layers, 4, cfg.model.ffn_dropout, time_aware=True)

    def forward(self, batch, grl_coef=1.0):
        H, M = self.bank(batch), self.masks(batch)
        toks, masks = [], []
        for i, n in enumerate(self.names):
            h = self.uni[n](H[n], batch["time"], batch["seq_mask"]) + self.mod_emb[i]
            toks.append(h)
            masks.append(M[n] * batch["seq_mask"])
        x = torch.cat(toks, 1)
        mask = torch.cat(masks, 1)
        time = torch.cat([batch["time"]] * len(self.names), 1)
        h = self.joint(x, time, mask)
        z, _ = self.pool(h, mask)
        return self.finish(z, batch, grl_coef)


# --------------------------------------------------------------------------- DepAudioNet
class DepAudioNet(BaselineBase):
    """DepAudioNet：对数梅尔谱 -> 1D-CNN -> LSTM。每个话语的谱先经 CNN 得到向量，再用 LSTM 沿话语建模。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        b, n_mels = cfg.baseline, cfg.model.speech.n_mels
        self.cnn = nn.Sequential(nn.Conv1d(n_mels, b.channels, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(b.channels, b.channels, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.lstm = nn.LSTM(b.channels, b.lstm_hidden // 2, batch_first=True, bidirectional=True)
        self.to_d = nn.Linear(b.lstm_hidden, self.d)
        self.bank = nn.ModuleDict()                    # 不使用 EncoderBank
        self.names = ["A"]
        self.groups = {"A": "A"}

    def pretrained_parameters(self):
        return []

    def forward(self, batch, grl_coef=1.0):
        mel = batch["logmel"]                            # [B,n,Tm,F]
        B, n, Tm, Fm = mel.shape
        u = self.cnn(mel.reshape(B * n, Tm, Fm).transpose(1, 2)).reshape(B, n, -1)
        h = self.to_d(run_rnn(self.lstm, u, batch["seq_mask"]))
        valid = batch["seq_mask"] * batch["m"]["A"]
        z, _ = self.pool(h, torch.where(valid.sum(1, keepdim=True) > 0, valid, batch["seq_mask"]))
        return self.finish(z, batch, grl_coef)


# --------------------------------------------------------------------------- DMPF
class DMPF(BaselineBase):
    """解耦多视角融合：两个声学视角（wav2vec 2.0 与对数梅尔 CNN-GRU）各自解耦为共享表示与私有表示，
    以正交约束 ||S^T P||_F^2 抑制两者冗余，拼接后池化。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        b = cfg.baseline
        self.shared = nn.Linear(self.d, b.shared_dim)               # 两个视角共用的共享投影
        self.private = nn.ModuleDict({n: nn.Linear(self.d, b.shared_dim) for n in self.names})
        self.out = nn.Linear(b.shared_dim * (1 + len(self.names)), self.d)
        self.w = b.ortho_weight

    def forward(self, batch, grl_coef=1.0):
        H, M = self.bank(batch), self.masks(batch)
        sh = [F.normalize(self.shared(H[n]), dim=-1) for n in self.names]
        pr = [F.normalize(self.private[n](H[n]), dim=-1) for n in self.names]
        valid = batch["seq_mask"].unsqueeze(-1)
        ortho = 0.0
        for s in sh:
            for p in pr:
                ortho = ortho + ((s * valid).transpose(1, 2) @ (p * valid)).pow(2).sum() / valid.sum().clamp(min=1.0)
        h = self.out(torch.cat([torch.stack(sh, 0).mean(0)] + pr, -1))
        z, _ = self.pool(h, batch["seq_mask"])
        return self.finish(z, batch, grl_coef, aux_loss=self.w * ortho)


BASELINES = {"early_fusion": EarlyFusion, "late_fusion": LateFusion, "mlan": MLAN, "mult": MulT,
             "tamfn": TAMFN, "ttfnet": TTFNet, "depaudionet": DepAudioNet, "dmpf": DMPF}
