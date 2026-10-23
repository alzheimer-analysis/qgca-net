"""各语料协议共用的工具：索引读取、交叉验证、"只在训练数据内做模型选择"的统一流程。

被选择的决策：编码器、保真度指标权重、tau、λ_ent、p_md、注意力头数、训练轮数、风险头判决阈值。
评估集上的受试者不得影响其中任何一项。
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import update_cfg
from ..data.normalization import FeatureNormalizer
from ..engine.trainer import (aggregate_subjects, average_over_runs, build_dataset, predict, train_model)
from ..evaluation import metrics as M
from ..evaluation import stats as S
from ..models.factory import prepare_model_cfg
from ..utils import get_logger, load_json

log = get_logger()


# ------------------------------------------------------------------------------ 索引
def load_index(processed_dir: str) -> List[Dict]:
    """processed_dir/index.json: [{record_id, subject_id, path, split, label, score, age, age_group, task}, ...]"""
    idx = load_json(os.path.join(processed_dir, "index.json"))
    for e in idx:
        if not os.path.isabs(e["path"]):
            e["path"] = os.path.join(processed_dir, e["path"])
    return idx


def select_entries(index: List[Dict], splits: Optional[Sequence[str]] = None,
                   subjects: Optional[Sequence[str]] = None) -> List[Dict]:
    sub = set(subjects) if subjects is not None else None
    return [e for e in index if (splits is None or e["split"] in splits) and (sub is None or e["subject_id"] in sub)]


def subject_table(entries: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for e in entries:
        out.setdefault(e["subject_id"], e)
    return out


def paths_of(entries: List[Dict]) -> List[str]:
    return [e["path"] for e in entries]


def fit_normalizer(entries: List[Dict], cfg) -> FeatureNormalizer:
    return FeatureNormalizer().fit(paths_of(entries), use_app=bool(cfg.model.video.use_app))


# ------------------------------------------------------------------------------ 训练+预测
def seeds_of(cfg, selection: bool = False) -> List[int]:
    if selection and cfg.get("selection_seeds"):
        return list(cfg.selection_seeds)
    return list(cfg.seed_list)


def train_and_predict(cfg, train_entries: List[Dict], eval_sets: Dict[str, List[Dict]], epochs: int,
                      seeds: Sequence[int], normalizer: Optional[FeatureNormalizer] = None
                      ) -> Dict[str, List[Dict]]:
    """用若干种子训练，返回 {eval_name: 种子平均后的录音级预测}。cfg 需已 prepare_model_cfg。"""
    normalizer = normalizer or fit_normalizer(train_entries, cfg)
    runs: Dict[str, List[List[Dict]]] = {k: [] for k in eval_sets}
    for seed in seeds:
        train_ds = build_dataset(paths_of(train_entries), normalizer, cfg, train=True, seed=seed)
        model, device = train_model(cfg, train_ds, epochs, seed)
        for name, entries in eval_sets.items():
            ev = build_dataset(paths_of(entries), normalizer, cfg, train=False)
            runs[name].append(predict(model, ev, cfg, device))
        del model
    return {k: average_over_runs(v) for k, v in runs.items()}


# ------------------------------------------------------------------------------ 交叉验证与选择
def metric_for_selection(rows: List[Dict], task: str):
    """返回 (score, threshold)。task=cls: 折外宏 F1（在折外最优阈值处）；task=reg: -RMSE。"""
    if task == "reg":                       # 回归（AVEC2014）按录音计算 RMSE
        y = np.array([r["score"] for r in rows])
        p = np.array([r["reg"] for r in rows])
        ok = np.isfinite(y)
        return -M.rmse(y[ok], p[ok]), 0.5
    subj = aggregate_subjects(rows)
    y = np.array([s["y"] for s in subj]).astype(int)
    prob = np.array([s["prob"] for s in subj])
    thr = M.best_threshold(y, prob)
    return M.macro_f1(y, (prob >= thr).astype(int)), thr


def cv_oof(cfg, entries: List[Dict], folds, epochs: int, seeds: Sequence[int]) -> List[Dict]:
    """给定 (train_subjects, val_subjects) 折，返回全部折外录音级预测（种子平均）。"""
    oof = []
    for tr_s, va_s in folds:
        tr = select_entries(entries, subjects=tr_s)
        va = select_entries(entries, subjects=va_s)
        oof += train_and_predict(cfg, tr, {"val": va}, epochs, seeds)["val"]
    return oof


def select_epochs_threshold(cfg, entries: List[Dict], folds, task: str, seeds: Optional[Sequence[int]] = None
                            ) -> Dict:
    """在训练数据的交叉验证上选择训练轮数与判决阈值（并为晚期融合给出单模态加权）。"""
    seeds = seeds or seeds_of(cfg, selection=True)
    best = None
    table = []
    for ep in cfg.train.epochs_grid:
        oof = cv_oof(cfg, entries, folds, int(ep), seeds)
        score, thr = metric_for_selection(oof, task)
        table.append({"epochs": int(ep), "score": float(score), "threshold": float(thr)})
        log.info("selection: epochs=%d score=%.4f thr=%.2f", ep, score, thr)
        if best is None or score > best["score"]:
            best = {"epochs": int(ep), "score": float(score), "threshold": float(thr), "oof": oof}
    best["table"] = table
    if cfg.model.name == "late_fusion":
        subj = aggregate_subjects(best["oof"])
        by = {r["record_id"]: r for r in best["oof"]}
        # 单模态验证宏 F1 作为投票权重
        K = len(best["oof"][0]["branch_prob"])
        sid2rows: Dict[str, List[Dict]] = {}
        for r in best["oof"]:
            sid2rows.setdefault(r["subject_id"], []).append(r)
        y = np.array([sid2rows[s["subject_id"]][0]["y"] for s in subj]).astype(int)
        f1s = []
        for k in range(K):
            pk = np.array([np.mean([r["branch_prob"][k] for r in sid2rows[s["subject_id"]]]) for s in subj])
            f1s.append(M.macro_f1(y, (pk >= 0.5).astype(int)))
        best["late_weights"] = f1s
    best.pop("oof", None)
    return best


def cv_score(cfg, entries: List[Dict], folds, task: str, epochs: int, seeds: Sequence[int]) -> float:
    """超参数搜索的单配置得分（折外）。"""
    return metric_for_selection(cv_oof(cfg, entries, folds, epochs, seeds), task)[0]


def hyperparameter_search(cfg, entries, folds, task: str, candidates: List[Dict], epochs: int,
                          seeds: Optional[Sequence[int]] = None) -> Dict:
    """在训练数据内对一组候选配置补丁（dotted-key dict）打分，返回最优补丁。"""
    seeds = seeds or seeds_of(cfg, selection=True)
    results = []
    for i, patch in enumerate(candidates):
        c = prepare_model_cfg(update_cfg(cfg, patch))
        s = cv_score(c, entries, folds, task, epochs, seeds)
        results.append({"patch": patch, "score": float(s)})
        log.info("search %d/%d score=%.4f patch=%s", i + 1, len(candidates), s, patch)
    best = max(results, key=lambda r: r["score"])
    return {"best": best, "all": results}


def sample_candidates(space: Dict, n: int, seed: int) -> List[Dict]:
    """从离散搜索空间里无放回地随机抽取 n 个配置（若空间更小则取全部）。"""
    import itertools
    keys = list(space.keys())
    grid = list(itertools.product(*[space[k] for k in keys]))
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(grid))[:n]
    return [dict(zip(keys, grid[i])) for i in pick]


# ------------------------------------------------------------------------------ 评估汇总
def eval_classification(rows: List[Dict], threshold: float, n_boot: int = 10000, seed: int = 0) -> Dict:
    subj = aggregate_subjects(rows)
    y = np.array([s["y"] for s in subj]).astype(int)
    prob = np.array([s["prob"] for s in subj])
    m = M.classification_metrics(y, prob, threshold)
    f1, lo, hi = S.macro_f1_ci(y, prob, threshold, n_boot, seed)
    m["macro_f1_ci"] = [lo, hi]
    m["auc_ci"] = list(S.hanley_mcneil_ci(m["auc"], int(y.sum()), int((1 - y).sum()))) if np.isfinite(m["auc"]) else None
    m["n"], m["n_pos"], m["threshold"] = len(y), int(y.sum()), float(threshold)
    sc = np.array([s["score"] for s in subj])
    reg = np.array([s["reg"] for s in subj])
    ok = np.isfinite(sc)
    if ok.any():
        m["rmse"], m["mae"] = M.rmse(sc[ok], reg[ok]), M.mae(sc[ok], reg[ok])
        m["rmse_ci"] = list(S.rmse_ci(sc[ok], reg[ok], y[ok], n_boot, seed)[1:])
    return m


def eval_regression(rows: List[Dict], n_boot: int = 10000, seed: int = 0, by_recording: bool = True) -> Dict:
    """AVEC2014：RMSE/MAE/CCC 在录音上计算；RMSE 的 CI 按受试者（而非录音）重采样。"""
    y = np.array([r["score"] for r in rows])
    p = np.array([r["reg"] for r in rows])
    sid = np.array([r["subject_id"] for r in rows])
    out = M.regression_metrics(y, p)
    uniq = np.unique(sid)
    members = [np.flatnonzero(sid == s) for s in uniq]
    f = lambda i: M.rmse(y[np.concatenate([members[j] for j in i])], p[np.concatenate([members[j] for j in i])])
    _, lo, hi = S.bootstrap_ci(f, len(uniq), np.zeros(len(uniq)), n_boot, seed)
    out["rmse_ci"] = [lo, hi]
    out["n_recordings"], out["n_subjects"] = len(rows), len(uniq)
    return out
