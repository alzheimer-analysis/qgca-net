"""缺失/损坏模态的鲁棒性评估（DAIC-WOZ 留出集）。

对已训练模型施加：整模态缺失（T/A/V）、30% 视频帧遮挡、语音加白噪声（SNR 10 dB）。
模型在干净训练数据上训练一次（种子集合同主实验），在各条件下复用，保证条件间配对可比。
遮挡/加噪后的特征由 scripts/make_perturbed_records.py 预先生成到独立目录（含 index.json）。
"""
import os
from typing import Dict, List

import numpy as np

from ..config import update_cfg
from ..engine.trainer import average_over_runs, build_dataset, predict, train_model
from ..models.factory import prepare_model_cfg
from ..utils import get_logger, load_json, save_json
from . import common as C

log = get_logger()


def run_robustness(cfg, rob_cfg: Dict, perturbed_dirs: Dict[str, str], selection: Dict, out_dir: str) -> Dict:
    """rob_cfg: configs/robustness.yaml 的 conditions；perturbed_dirs: {condition: 扰动后 processed 目录}。"""
    cfg = prepare_model_cfg(cfg)
    index = C.load_index(cfg.data.processed_dir)
    train = C.select_entries(index, ["train"])
    heldout = C.select_entries(index, cfg.protocol.eval_splits)
    norm = C.fit_normalizer(train, cfg)
    ids = {e["record_id"] for e in heldout}

    runs: Dict[str, List[List[Dict]]] = {k: [] for k in rob_cfg["conditions"]}
    for seed in C.seeds_of(cfg):
        model, device = train_model(cfg, build_dataset(C.paths_of(train), norm, cfg, True, seed=seed),
                                    selection["epochs"], seed)
        for name, spec in rob_cfg["conditions"].items():
            kind = spec.get("kind", "clean")
            ccfg, entries = cfg, heldout
            if kind == "drop_modality":
                ccfg = update_cfg(cfg, {"eval.drop_modalities": [spec["group"]]})
            elif kind in ("video_occlusion", "speech_noise"):
                pidx = C.load_index(perturbed_dirs[name])
                entries = [e for e in pidx if e["record_id"] in ids]
            ev = build_dataset(C.paths_of(entries), norm, ccfg, train=False)
            runs[name].append(predict(model, ev, ccfg, device))
        del model

    result = {}
    for name, rr in runs.items():
        rows = average_over_runs(rr)
        result[name] = C.eval_classification(rows, selection["threshold"], cfg.eval.bootstrap)
        save_json(rows, os.path.join(out_dir, f"predictions_{name}.json"))
    save_json(result, os.path.join(out_dir, "metrics.json"))
    return result
