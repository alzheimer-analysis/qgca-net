"""训练目标：L = λ_cls·L_cls + λ_reg·L_reg + λ_ent·L_ent + λ_adv·L_adv"""
from typing import Dict

import torch
import torch.nn.functional as F


def focal_loss(logit: torch.Tensor, y: torch.Tensor, gamma: float = 2.0, alpha_pos: float = 0.7) -> torch.Tensor:
    """L_cls = -α_t (1-p_t)^γ log p_t；logit:[B]，y∈{0,1}。逐样本返回。"""
    logit = logit.float()
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0.5, p, 1 - p)
    alpha_t = torch.where(y > 0.5, torch.full_like(p, alpha_pos), torch.full_like(p, 1 - alpha_pos))
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")      # = -log p_t
    return alpha_t * (1 - pt).pow(gamma) * ce


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """|e|<=δ: ½e²；否则 δ(|e| - ½δ)；在 [0,1] 归一化的尺度上计算。逐样本返回。"""
    e = (target - pred.float()).abs()
    return torch.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta))


class QGCALoss:
    def __init__(self, cfg):
        self.l = cfg.loss
        self.task = cfg.model.task

    def __call__(self, out: Dict[str, torch.Tensor], batch: Dict, domain_target: torch.Tensor = None
                 ) -> Dict[str, torch.Tensor]:
        y, r, has_reg = batch["y"], batch["r"], batch["has_reg"]
        has_cls = (y >= 0).float()
        zero = out["logit"].new_zeros(()).float()

        l_cls = zero
        if self.l.lambda_cls > 0 and has_cls.sum() > 0:
            per = focal_loss(out["logit"], y.clamp(min=0), self.l.focal_gamma, self.l.focal_alpha_pos)
            l_cls = (per * has_cls).sum() / has_cls.sum()
        l_reg = zero
        if self.l.lambda_reg > 0 and has_reg.sum() > 0:
            per = huber_loss(out["reg"], r, self.l.huber_delta)
            l_reg = (per * has_reg).sum() / has_reg.sum()
        l_ent = out["ent_loss"].float()
        l_adv = zero
        if self.l.lambda_adv > 0 and domain_target is not None:
            l_adv = F.cross_entropy(out["domain_logits"].float(), domain_target)

        total = (self.l.lambda_cls * l_cls + self.l.lambda_reg * l_reg
                 + self.l.lambda_ent * l_ent + self.l.lambda_adv * l_adv)
        extra = out.get("aux_loss")
        if extra is not None:                                                 # 例如 DMPF 的正交约束
            total = total + extra.float()
        return {"total": total, "cls": l_cls.detach(), "reg": l_reg.detach(),
                "ent": l_ent.detach(), "adv": l_adv.detach()}
