"""DAIC-WOZ 协议。

所有决策在 107 个训练会话的分层 5 折交叉验证中完成；选定配置在全部训练会话上用 CV 选出的 epoch 数重训，
并在留出集 H（dev + test，82 例，26 例抑郁）上只评估一次；另外报告仅 dev（35 例）的结果。
"""
import os
from typing import Dict

import numpy as np

from ..config import update_cfg
from ..data.splits import subject_stratified_kfold
from ..models.factory import prepare_model_cfg
from ..utils import get_logger, save_json
from . import common as C

log = get_logger()


def daic_folds(train_entries, k: int, seed: int = 0):
    tab = C.subject_table(train_entries)
    subs = list(tab.keys())
    return subject_stratified_kfold(subs, [tab[s]["label"] for s in subs], k, seed)


def run_daic(cfg, out_dir: str, selection: Dict = None) -> Dict:
    cfg = prepare_model_cfg(cfg)
    index = C.load_index(cfg.data.processed_dir)
    train = C.select_entries(index, ["train"])
    heldout = C.select_entries(index, cfg.protocol.eval_splits)
    dev = C.select_entries(index, ["dev"])
    log.info("DAIC-WOZ: train=%d held-out=%d (dev=%d)", len(train), len(heldout), len(dev))

    # ---- 1) 训练集 5 折 CV：训练轮数 + 判决阈值（以及晚期融合的投票权重）----
    if selection is None:
        folds = daic_folds(train, cfg.protocol.cv_folds)
        selection = C.select_epochs_threshold(cfg, train, folds, task="cls")
        save_json(selection, os.path.join(out_dir, "selection.json"))
    if "late_weights" in selection:
        cfg = update_cfg(cfg, {"model.late_weights": selection["late_weights"]})

    # ---- 2) 在全部训练会话上重训，留出集只评估一次 ----
    preds = C.train_and_predict(cfg, train, {"heldout": heldout}, selection["epochs"], C.seeds_of(cfg))
    rows = preds["heldout"]
    thr = selection["threshold"]
    result = {"selection": selection, "heldout": C.eval_classification(rows, thr, cfg.eval.bootstrap)}
    if cfg.protocol.report_dev_only:
        dev_ids = {e["record_id"] for e in dev}
        result["dev_only"] = C.eval_classification([r for r in rows if r["record_id"] in dev_ids], thr, cfg.eval.bootstrap)
    save_json(rows, os.path.join(out_dir, "predictions_heldout.json"))
    save_json(result, os.path.join(out_dir, "metrics.json"))
    return result
