"""训练与推理。

AdamW（权重衰减 0.01）；预训练编码层学习率 1e-5，新初始化模块 1e-3；batch 16；5 个 epoch 线性预热后余弦退火；
混合精度。训练样本可以由多个数据集拼接（跨语料训练，附带域对抗分支），可含无标签样本（仅供域判别器使用）。
"""
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..data.dataset import Collator, DepressionDataset
from ..data.normalization import FeatureNormalizer
from ..data.schema import SCORE_MAX
from ..losses import QGCALoss
from ..models.factory import build_model, prepare_model_cfg
from ..utils import get_device, get_logger, set_seed, to_device

log = get_logger()


def make_loader(dataset: Dataset, cfg, shuffle: bool, batch_size: Optional[int] = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size or cfg.train.batch_size, shuffle=shuffle,
                      num_workers=cfg.data.num_workers, collate_fn=Collator(cfg), drop_last=False,
                      persistent_workers=False, pin_memory=torch.cuda.is_available())


def build_dataset(paths: Sequence[str], normalizer: FeatureNormalizer, cfg, train: bool, labeled: bool = True,
                  seed: int = 0) -> DepressionDataset:
    return DepressionDataset(list(paths), normalizer, cfg, train=train, labeled=labeled, seed=seed)


def _param_groups(model: torch.nn.Module, cfg):
    pre = {id(p) for p in model.pretrained_parameters()}
    pre_params = [p for p in model.parameters() if id(p) in pre and p.requires_grad]
    new_params = [p for p in model.parameters() if id(p) not in pre and p.requires_grad]
    groups = [{"params": new_params, "lr": cfg.train.lr_new}]
    if pre_params:
        groups.append({"params": pre_params, "lr": cfg.train.lr_pretrained})
    return groups


def _lr_lambda(total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


def train_model(cfg, train_set: Dataset, epochs: int, seed: int, model: Optional[torch.nn.Module] = None,
                epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
                lr_scale: float = 1.0):
    """训练一个模型。cfg 需已经过 prepare_model_cfg。返回 (model, device)。

    lr_scale 用于微调阶段缩小学习率；model 不为 None 时在其基础上继续训练（迁移微调）。
    """
    set_seed(seed)
    device = get_device(cfg.device)
    if model is None:
        model, _ = build_model(cfg)
    model.to(device)
    loader = make_loader(train_set, cfg, shuffle=True)
    groups = _param_groups(model, cfg)
    for g in groups:
        g["lr"] *= lr_scale
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    total = epochs * len(loader)
    warm = min(cfg.train.warmup_epochs, max(1, epochs // 4)) * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, warm))
    criterion = QGCALoss(cfg)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for ep in range(epochs):
        model.train()
        run = {"total": 0.0, "cls": 0.0, "reg": 0.0, "ent": 0.0, "adv": 0.0}
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch)
                losses = criterion(out, batch, domain_target=batch["dataset_id"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k in run:
                run[k] += float(losses[k])
        log.info("seed %d epoch %d/%d " % (seed, ep + 1, epochs)
                 + " ".join(f"{k}={v / max(1, len(loader)):.4f}" for k, v in run.items()))
        if epoch_callback is not None:
            epoch_callback(ep + 1, model)
    return model, device


@torch.no_grad()
def predict(model: torch.nn.Module, dataset: Dataset, cfg, device=None, head_override: Optional[str] = None
            ) -> List[Dict]:
    """逐录音预测。返回 [{record_id, subject_id, dataset, prob, reg(原始量表分), y, score}]。

    head_override: 使用另一个数据集的严重度头做预测（零样本迁移时用源域头），得分仍按目标数据集量表换算。
    """
    device = device or get_device(cfg.device)
    model.to(device).eval()
    loader = make_loader(dataset, cfg, shuffle=False, batch_size=max(1, cfg.train.batch_size))
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    rows = []
    for batch in loader:
        b = to_device(batch, device)
        if head_override is not None:
            b["dataset"] = [head_override] * len(b["dataset"])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(b)
        prob = torch.sigmoid(out["logit"].float()).cpu().numpy()
        reg = out["reg"].float().cpu().numpy()
        branch = torch.sigmoid(out["branch_logits"].float()).cpu().numpy() if "branch_logits" in out else None
        for i in range(len(prob)):
            ds = batch["dataset"][i]
            rows.append({"branch_prob": branch[i].tolist() if branch is not None else None,"record_id": batch["record_id"][i], "subject_id": batch["subject_id"][i], "dataset": ds,
                         "prob": float(prob[i]), "reg": float(np.clip(reg[i], 0.0, 1.0) * SCORE_MAX[ds]),
                         "y": float(batch["y"][i]), "score": float(batch["r"][i]) * SCORE_MAX[ds]
                         if float(batch["has_reg"][i]) > 0 else float("nan")})
    return rows


def aggregate_subjects(rows: List[Dict]) -> List[Dict]:
    """录音级 -> 受试者级：风险概率与严重度得分取各录音的平均（MPSC 六段、AVEC 多段）。"""
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(r["subject_id"], []).append(r)
    out = []
    for sid, rs in by.items():
        out.append({"subject_id": sid, "dataset": rs[0]["dataset"], "prob": float(np.mean([x["prob"] for x in rs])),
                    "reg": float(np.mean([x["reg"] for x in rs])), "y": rs[0]["y"],
                    "score": float(np.nanmean([x["score"] for x in rs])) if not all(np.isnan(x["score"]) for x in rs)
                    else float("nan"), "n_rec": len(rs)})
    return out


def average_over_runs(runs: List[List[Dict]], key: str = "record_id") -> List[Dict]:
    """对多个种子（或重复）的预测按 key 取平均。"""
    acc: Dict[str, Dict] = {}
    cnt: Dict[str, int] = {}
    for rows in runs:
        for r in rows:
            k = r[key]
            if k not in acc:
                acc[k] = dict(r)
                cnt[k] = 1
            else:
                acc[k]["prob"] += r["prob"]
                acc[k]["reg"] += r["reg"]
                if r.get("branch_prob") is not None:
                    acc[k]["branch_prob"] = [a + b for a, b in zip(acc[k]["branch_prob"], r["branch_prob"])]
                cnt[k] += 1
    for k in acc:
        acc[k]["prob"] /= cnt[k]
        acc[k]["reg"] /= cnt[k]
        if acc[k].get("branch_prob") is not None:
            acc[k]["branch_prob"] = [a / cnt[k] for a in acc[k]["branch_prob"]]
    return list(acc.values())


def concat_datasets(parts: Sequence[Dataset]) -> Dataset:
    return parts[0] if len(parts) == 1 else ConcatDataset(list(parts))
