"""注意力池化、风险/严重度多任务头与域对抗分支（梯度反转层）。"""
import torch
import torch.nn as nn

from ..data.schema import DATASET_IDS


class AttentivePooling(nn.Module):
    """e_i = w^T tanh(W z_i + b), α = softmax_i(e_i) 仅在有效位置上归一化, z = Σ α_i z_i"""

    def __init__(self, d: int, mode: str = "attentive"):
        super().__init__()
        self.mode = mode
        if mode == "attentive":
            self.proj = nn.Linear(d, d)
            self.w = nn.Linear(d, 1, bias=False)

    def forward(self, z: torch.Tensor, valid: torch.Tensor):
        if self.mode == "mean":
            a = valid / valid.sum(1, keepdim=True).clamp(min=1.0)
        else:
            e = self.w(torch.tanh(self.proj(z))).squeeze(-1).float().masked_fill(valid <= 0, -1e4)
            a = torch.softmax(e, 1) * valid
            a = a / a.sum(1, keepdim=True).clamp(min=1e-8)
        return (a.unsqueeze(-1).to(z.dtype) * z).sum(1), a


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef):
        ctx.coef = coef
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coef * g, None


def grad_reverse(x, coef: float = 1.0):
    return _GradReverse.apply(x, coef)


class MultiTaskHeads(nn.Module):
    """风险头 ŷ = σ(w_c^T z + b_c)；每个数据集独立的线性严重度头 r̂_d = w_d^T z + b_d；可选域判别器。"""

    def __init__(self, d: int, datasets=("daic_woz", "avec2014", "mpsc"), n_domains: int = 3, dropout: float = 0.1,
                 shared_severity: bool = False):
        """shared_severity=True 时所有数据集共用一个严重度头（迁移实验中"无数据集专属头"的对照）。"""
        super().__init__()
        self.risk = nn.Linear(d, 1)
        self.shared = shared_severity
        keys = ["shared"] if shared_severity else list(datasets)
        self.severity = nn.ModuleDict({k: nn.Linear(d, 1) for k in keys})
        self.domain = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_domains))
        self.datasets = list(datasets)

    def forward(self, z: torch.Tensor, dataset_names, grl_coef: float = 1.0):
        logit = self.risk(z).squeeze(-1)
        reg = z.new_zeros(z.size(0))
        names = ["shared"] * len(dataset_names) if self.shared else list(dataset_names)
        for ds in set(names):
            idx = torch.tensor([i for i, x in enumerate(names) if x == ds], device=z.device)
            reg = reg.index_put((idx,), self.severity[ds](z[idx]).squeeze(-1).to(reg.dtype))
        dom = self.domain(grad_reverse(z, grl_coef))
        return logit, reg, dom
