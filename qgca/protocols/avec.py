"""AVEC2014 协议。

所有决策在官方 dev 分区上做（训练集训练、dev 评估）；选定配置在 train+dev 上重训，在 100 条测试录音上只评估一次。
文本分支整体被掩码（无转写）。回归指标 RMSE/MAE/CCC；RMSE 的 CI 按受试者重采样。
"""
import os
from typing import Dict

import numpy as np

from ..engine.trainer import aggregate_subjects
from ..evaluation import metrics as M
from ..models.factory import prepare_model_cfg
from ..utils import get_logger, save_json
from . import common as C

log = get_logger()


def select_on_dev(cfg, train, dev) -> Dict:
    """在 dev 分区上选择训练轮数（最小 RMSE）与风险阈值（BDI-II>=14 的头，仅供迁移实验使用）。"""
    best, table = None, []
    for ep in cfg.train.epochs_grid:
        rows = C.train_and_predict(cfg, train, {"dev": dev}, int(ep), C.seeds_of(cfg, selection=True))["dev"]
        score, _ = C.metric_for_selection(rows, "reg")
        subj = aggregate_subjects(rows)
        y = np.array([s["y"] for s in subj]).astype(int)
        thr = M.best_threshold(y, np.array([s["prob"] for s in subj])) if len(set(y.tolist())) > 1 else 0.5
        table.append({"epochs": int(ep), "score": float(score), "threshold": float(thr)})
        log.info("AVEC dev selection epochs=%d -RMSE=%.4f", ep, score)
        if best is None or score > best["score"]:
            best = {"epochs": int(ep), "score": float(score), "threshold": float(thr)}
    best["table"] = table
    return best


def run_avec(cfg, out_dir: str, selection: Dict = None) -> Dict:
    cfg = prepare_model_cfg(cfg)
    p = cfg.protocol
    index = C.load_index(cfg.data.processed_dir)
    train = C.select_entries(index, p.train_splits)
    dev = C.select_entries(index, [p.select_split])
    final_train = C.select_entries(index, p.final_train_splits)
    test = C.select_entries(index, [p.test_split])
    log.info("AVEC2014: train=%d dev=%d test=%d recordings", len(train), len(dev), len(test))

    if selection is None:
        selection = select_on_dev(cfg, train, dev)
        save_json(selection, os.path.join(out_dir, "selection.json"))

    rows = C.train_and_predict(cfg, final_train, {"test": test}, selection["epochs"], C.seeds_of(cfg))["test"]
    result = {"selection": selection, "test": C.eval_regression(rows, cfg.eval.bootstrap)}
    save_json(rows, os.path.join(out_dir, "predictions_test.json"))
    save_json(result, os.path.join(out_dir, "metrics.json"))
    return result
