"""迁移实验：跨语料（AVEC2014 <-> DAIC-WOZ）与跨语言/青少年（DAIC-WOZ -> MPSC）。

适配组件:
  DSH 数据集专属严重度头（关闭时所有数据集共用一个头）
  DAL 域对抗学习（GRL + 域判别器，λ_adv = 0.1）

零样本: 目标域无标签。DAL 仅使用目标域训练录音的"输入"（无标签）训练域判别器。
微调  : 在源模型基础上用目标域训练集（有标签）继续训练，学习率缩小为原来的 1/10；DAL 时同时输入源域样本（仅域标签）。
跨语料中目标域文本被掩码（AVEC2014 无转写），视频仅使用两库共享的 OpenFace 特征。
"""
import copy
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from ..config import update_cfg
from ..data.splits import mpsc_nested_splits
from ..engine.trainer import (average_over_runs, build_dataset, concat_datasets, predict, train_model)
from ..models.factory import build_model, prepare_model_cfg
from ..utils import get_device, get_logger, save_json
from . import common as C

log = get_logger()


def transfer_cfg(cfg, dsh: bool, dal: bool, xlingual: bool = False):
    patch = {"model.video.use_app": False, "model.shared_severity": (not dsh),
             "loss.lambda_adv": 0.1 if dal else 0.0}
    if xlingual:
        patch["model.text.backbone"] = "xlm-roberta-base"
    return prepare_model_cfg(update_cfg(cfg, patch))


def _train_source(cfg, src_entries, tgt_entries, epochs: int, seed: int, norm_src, norm_tgt):
    """源域有标签 + （DAL 时）目标域无标签样本。"""
    parts = [build_dataset(C.paths_of(src_entries), norm_src, cfg, True, labeled=True, seed=seed)]
    if cfg.loss.lambda_adv > 0:
        parts.append(build_dataset(C.paths_of(tgt_entries), norm_tgt, cfg, True, labeled=False, seed=seed))
    return train_model(cfg, concat_datasets(parts), epochs, seed)


def _finetune(cfg, model, src_entries, tgt_entries, epochs: int, seed: int, norm_src, norm_tgt):
    parts = [build_dataset(C.paths_of(tgt_entries), norm_tgt, cfg, True, labeled=True, seed=seed)]
    if cfg.loss.lambda_adv > 0:
        parts.append(build_dataset(C.paths_of(src_entries), norm_src, cfg, True, labeled=False, seed=seed))
    return train_model(cfg, concat_datasets(parts), epochs, seed, model=model, lr_scale=0.1)


def run_cross_corpus(base_cfgs: Dict[str, object], direction: str, dsh: bool, dal: bool, setting_epochs: Dict,
                     out_dir: str) -> Dict:
    """direction: avec_to_daic | daic_to_avec。base_cfgs = {'daic_woz': cfg, 'avec2014': cfg}（各自协议配置）。
    setting_epochs: {'source': 源域训练轮数, 'finetune': 微调轮数, 'threshold': 风险阈值}。"""
    src_name, tgt_name = ("avec2014", "daic_woz") if direction == "avec_to_daic" else ("daic_woz", "avec2014")
    scfg, tcfg = base_cfgs[src_name], base_cfgs[tgt_name]
    cfg = transfer_cfg(scfg, dsh, dal)
    cfg_t = transfer_cfg(tcfg, dsh, dal)
    s_idx, t_idx = C.load_index(scfg.data.processed_dir), C.load_index(tcfg.data.processed_dir)
    src_train = C.select_entries(s_idx, scfg.protocol.get("final_train_splits", ["train"]))
    tgt_train = C.select_entries(t_idx, tcfg.protocol.get("final_train_splits", ["train"]))
    tgt_eval = C.select_entries(t_idx, tcfg.protocol.get("eval_splits", [tcfg.protocol.get("test_split", "test")]))
    n_src, n_tgt = C.fit_normalizer(src_train, cfg), C.fit_normalizer(tgt_train, cfg_t)
    task = "cls" if tgt_name == "daic_woz" else "reg"

    zs_runs, ft_runs = [], []
    for seed in C.seeds_of(cfg):
        model, device = _train_source(cfg, src_train, tgt_train, setting_epochs["source"], seed, n_src, n_tgt)
        ev = build_dataset(C.paths_of(tgt_eval), n_tgt, cfg_t, train=False)
        zs_runs.append(predict(model, ev, cfg_t, device, head_override=src_name))        # 零样本: 源域头
        model, _ = _finetune(cfg_t, model, src_train, tgt_train, setting_epochs["finetune"], seed, n_src, n_tgt)
        ft_runs.append(predict(model, ev, cfg_t, device))
        del model

    res = {}
    for name, runs in (("zero_shot", zs_runs), ("fine_tuned", ft_runs)):
        rows = average_over_runs(runs)
        res[name] = (C.eval_classification(rows, setting_epochs["threshold"], cfg.eval.bootstrap) if task == "cls"
                     else C.eval_regression(rows, cfg.eval.bootstrap))
        save_json(rows, os.path.join(out_dir, f"predictions_{name}.json"))
    save_json(res, os.path.join(out_dir, "metrics.json"))
    return res


def run_cross_lingual(daic_cfg, mpsc_cfg, method_cfg_patch: Dict, dsh: bool, dal: bool, setting: Dict,
                      out_dir: str, only_repeat: Optional[int] = None) -> Dict:
    """DAIC-WOZ -> MPSC。零样本使用全部 100 名 MPSC 受试者；微调沿用 MPSC 嵌套 CV 外层折，
    因此域内、微调结果对应同一批测试者。method_cfg_patch 指定对比方法（如 model.name=mult）。"""
    dcfg = transfer_cfg(update_cfg(daic_cfg, method_cfg_patch), dsh, dal, xlingual=True)
    mcfg = transfer_cfg(update_cfg(mpsc_cfg, method_cfg_patch), dsh, dal, xlingual=True)
    d_idx, m_idx = C.load_index(dcfg.data.processed_dir), C.load_index(mcfg.data.processed_dir)
    d_train = C.select_entries(d_idx, ["train"])
    m_all = C.select_entries(m_idx)
    n_src, n_tgt_all = C.fit_normalizer(d_train, dcfg), C.fit_normalizer(m_all, mcfg)   # 零样本: 目标域归一化统计量取自无标签输入

    tab = C.subject_table(m_idx)
    subs = sorted(tab)
    splits = mpsc_nested_splits(subs, [tab[s]["label"] for s in subs], [tab[s]["age_group"] for s in subs],
                                mcfg.protocol.outer_folds, mcfg.protocol.outer_repeats, mcfg.protocol.inner_folds)
    zs_runs: List[List[Dict]] = []
    ft_rows: List[List[Dict]] = [[] for _ in range(mcfg.protocol.outer_repeats)]
    for seed in C.seeds_of(dcfg):
        model, device = _train_source(dcfg, d_train, m_all, setting["source_epochs"], seed, n_src, n_tgt_all)
        ev = build_dataset(C.paths_of(m_all), n_tgt_all, mcfg, train=False)
        zs_runs.append(predict(model, ev, mcfg, device, head_override="daic_woz"))
        state = copy.deepcopy(model.state_dict())
        for sp in splits:
            if only_repeat is not None and sp["repeat"] != only_repeat:
                continue
            tr = C.select_entries(m_idx, subjects=sp["train"])
            te = C.select_entries(m_idx, subjects=sp["test"])
            n_tr = C.fit_normalizer(tr, mcfg)
            model.load_state_dict(state)
            model, _ = _finetune(mcfg, model, d_train, tr, setting["finetune_epochs"], seed, n_src, n_tr)
            ft_rows[sp["repeat"]] += predict(model, build_dataset(C.paths_of(te), n_tr, mcfg, train=False), mcfg, device)
        del model

    res = {}
    thr = setting["threshold"]
    zs = average_over_runs(zs_runs)
    res["zero_shot"] = {"all": C.eval_classification(zs, thr, mcfg.eval.bootstrap)}
    ft = average_over_runs([r for r in ft_rows if r]) if any(ft_rows) else []
    lo, hi = mcfg.protocol.adolescent_age
    ado = {s for s, e in tab.items() if e.get("age") is not None and lo <= e["age"] <= hi}
    res["zero_shot"]["adolescent_13_19"] = C.eval_classification([r for r in zs if r["subject_id"] in ado], thr, mcfg.eval.bootstrap)
    if ft:
        res["fine_tuned"] = {"all": C.eval_classification(ft, thr, mcfg.eval.bootstrap),
                             "adolescent_13_19": C.eval_classification([r for r in ft if r["subject_id"] in ado], thr,
                                                                       mcfg.eval.bootstrap)}
    save_json(res, os.path.join(out_dir, "metrics.json"))
    return res
