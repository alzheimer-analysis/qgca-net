"""质量指标是否泄露标签信息（论文 Do the Quality Indicators Leak Label Information?）。

(1) 每个候选指标（受试者层面均值）与抑郁标签的点二列相关 r（DAIC-WOZ 训练集, n=107）；
(2) 仅以 q 的受试者级统计量为特征的逻辑回归分类器，在留出集上的 AUC 及 Hanley–McNeil CI。
行为类代理（发声比例、语音能量、回答长度、情感词率、头部姿态、视线）预期与标签相关；
保真度指标（7 个）预期 |r| 很小，且 q-only AUC 的置信区间包含 0.5。
"""
from typing import Dict, List

import numpy as np
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ..data.normalization import FeatureNormalizer
from ..data.quality import behavioral_quality, fidelity_quality
from ..data.schema import BEHAVIOR_NAMES, FIDELITY_NAMES, load_record
from .metrics import auc
from .stats import hanley_mcneil_ci


def _subject_matrix(paths: List[str]):
    """逐受试者（录音）聚合原始指标均值；NaN 列保持 NaN。"""
    rows, labels = [], []
    names = [f"{g}:{n}" for g in ("T", "A", "V") for n in FIDELITY_NAMES[g]] + BEHAVIOR_NAMES
    for p in paths:
        r = load_record(p)
        fid = np.concatenate([np.asarray(r["fid"][g], float) for g in ("T", "A", "V")], 1)
        beh = np.asarray(r["behav"], float)
        rows.append(np.concatenate([np.nanmean(fid, 0), np.nanmean(beh, 0)]))
        labels.append(int(r["label"]))
    return names, np.vstack(rows), np.asarray(labels)


def indicator_label_correlations(train_paths: List[str]) -> List[Dict]:
    names, X, y = _subject_matrix(train_paths)
    out = []
    for j, n in enumerate(names):
        ok = np.isfinite(X[:, j])
        if ok.sum() < 5 or np.std(X[ok, j]) < 1e-12:
            out.append({"indicator": n, "r": float("nan"), "p": float("nan"), "n": int(ok.sum())})
            continue
        r, p = sps.pointbiserialr(y[ok], X[ok, j])
        out.append({"indicator": n, "r": float(r), "p": float(p), "n": int(ok.sum())})
    return out


def _q_stats(paths: List[str], normalizer: FeatureNormalizer, qcfg, kind: str):
    feats, labels = [], []
    for p in paths:
        r = load_record(p)
        q = fidelity_quality(r, normalizer, qcfg) if kind == "fidelity" else behavioral_quality(r, normalizer, qcfg)
        f = []
        for g in ("T", "A", "V"):
            v = np.asarray(q[g], float)
            v = v[np.isfinite(v)] if np.isfinite(v).any() else np.ones(1)
            f += [v.mean(), v.std(), v.min(), v.max()]
        feats.append(f)
        labels.append(int(r["label"]))
    return np.asarray(feats), np.asarray(labels)


def q_only_auc(train_paths: List[str], eval_paths: List[str], normalizer: FeatureNormalizer, qcfg,
               kind: str = "fidelity", seed: int = 0) -> Dict[str, float]:
    """仅用 q 的受试者级统计量（均值/标准差/最小/最大）训练逻辑回归，返回留出集 AUC 与 95% CI。"""
    Xtr, ytr = _q_stats(train_paths, normalizer, qcfg, kind)
    Xev, yev = _q_stats(eval_paths, normalizer, qcfg, kind)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    clf.fit(Xtr, ytr)
    a = auc(yev, clf.predict_proba(Xev)[:, 1])
    lo, hi = hanley_mcneil_ci(a, int(yev.sum()), int((1 - yev).sum()))
    return {"auc": a, "ci_low": lo, "ci_high": hi, "kind": kind}
