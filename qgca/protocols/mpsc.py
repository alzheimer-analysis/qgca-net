"""MPSC 协议：说话人独立的重复嵌套交叉验证。

外层 5 折（按诊断与年龄组分层）用于评估，重复 5 次（不同随机划分）；每个外层训练集内部再做 4 折交叉验证，
所有决策（训练轮数、阈值等）只在内层完成。同一受试者的 6 段录音始终位于同一折；受试者得分取各段、各重复、
各种子的平均。青少年结果在外层测试折中 13–19 岁的受试者上计算。
"""
import os
from typing import Dict, List

import numpy as np

from ..config import update_cfg
from ..data.splits import mpsc_nested_splits
from ..engine.trainer import aggregate_subjects, average_over_runs
from ..models.factory import prepare_model_cfg
from ..utils import get_logger, save_json
from . import common as C

log = get_logger()


def run_mpsc(cfg, out_dir: str, only_repeat: int = None) -> Dict:
    cfg = prepare_model_cfg(cfg)
    p = cfg.protocol
    index = C.load_index(cfg.data.processed_dir)
    tab = C.subject_table(index)
    subs = sorted(tab.keys())
    splits = mpsc_nested_splits(subs, [tab[s]["label"] for s in subs], [tab[s]["age_group"] for s in subs],
                                p.outer_folds, p.outer_repeats, p.inner_folds, seed=0)
    per_repeat_rows: List[List[Dict]] = [[] for _ in range(p.outer_repeats)]
    thresholds = []
    for sp in splits:
        if only_repeat is not None and sp["repeat"] != only_repeat:
            continue
        tag = f"rep{sp['repeat']}_fold{sp['fold']}"
        train = C.select_entries(index, subjects=sp["train"])
        test = C.select_entries(index, subjects=sp["test"])
        sel_path = os.path.join(out_dir, f"selection_{tag}.json")
        sel = C.select_epochs_threshold(cfg, train, sp["inner"], task="cls")
        save_json(sel, sel_path)
        c = update_cfg(cfg, {"model.late_weights": sel["late_weights"]}) if "late_weights" in sel else cfg
        rows = C.train_and_predict(c, train, {"test": test}, sel["epochs"], C.seeds_of(cfg))["test"]
        per_repeat_rows[sp["repeat"]] += rows
        thresholds.append(sel["threshold"])
        log.info("MPSC %s done: epochs=%d thr=%.2f", tag, sel["epochs"], sel["threshold"])

    all_rows = [r for rep in per_repeat_rows for r in rep]
    thr = float(np.mean(thresholds))
    # 受试者得分：各段、各重复（以及已平均的种子）取平均
    merged = average_over_runs([rep for rep in per_repeat_rows if rep])
    result = {"all": C.eval_classification(merged, thr, cfg.eval.bootstrap)}
    lo, hi = p.adolescent_age
    ado = {s for s, e in tab.items() if e.get("age") is not None and lo <= e["age"] <= hi}
    result["adolescent_13_19"] = C.eval_classification([r for r in merged if r["subject_id"] in ado], thr,
                                                       cfg.eval.bootstrap)
    result["mean_threshold"] = thr
    save_json(all_rows, os.path.join(out_dir, "predictions_outer_test.json"))
    save_json(merged, os.path.join(out_dir, "predictions_merged.json"))
    save_json(result, os.path.join(out_dir, "metrics.json"))
    return result
