"""评价指标。

DAIC-WOZ : 抑郁类 precision / recall / F1、非抑郁类 F1、宏平均 F1（主指标）、AUC、PHQ-8 的 RMSE / MAE
AVEC2014 : RMSE、MAE、CCC
MPSC     : 宏平均 F1、AUC、灵敏度、特异度
校准     : Brier 分数与期望校准误差 ECE
"""
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def confusion(y: np.ndarray, pred: np.ndarray):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return tp, fp, fn, tn


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn, tn = confusion(y, pred)
    return 0.5 * (_f1(tp, fp, fn)[2] + _f1(tn, fn, fp)[2])


def auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def classification_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp, fp, fn, tn = confusion(y, pred)
    p, r, f1_pos = _f1(tp, fp, fn)
    _, _, f1_neg = _f1(tn, fn, fp)
    return {"precision": p, "recall": r, "f1_dep": f1_pos, "f1_nondep": f1_neg, "macro_f1": 0.5 * (f1_pos + f1_neg),
            "auc": auc(y, prob), "sensitivity": r, "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def ccc(y: np.ndarray, p: np.ndarray) -> float:
    """Lin 一致性相关系数。"""
    y, p = np.asarray(y, float), np.asarray(p, float)
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = ((y - my) * (p - mp)).mean()
    return float(2 * cov / (vy + vp + (my - mp) ** 2 + 1e-12))


def regression_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {"rmse": rmse(y, p), "mae": mae(y, p), "ccc": ccc(y, p)}


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(y)) ** 2))


def ece(y: np.ndarray, prob: np.ndarray, n_bins: int = 6, equal_count: bool = True) -> float:
    """期望校准误差（Guo et al., 2017）。equal_count=True 使用等样本量分箱。"""
    y, prob = np.asarray(y, float), np.asarray(prob, float)
    if equal_count:
        order = np.argsort(prob)
        bins = np.array_split(order, n_bins)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        ids = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
        bins = [np.where(ids == b)[0] for b in range(n_bins)]
    total, err = len(y), 0.0
    for b in bins:
        if len(b):
            err += len(b) / total * abs(y[b].mean() - prob[b].mean())
    return float(err)


def best_threshold(y: np.ndarray, prob: np.ndarray, grid=None) -> float:
    """在训练数据的（折外）预测上选使宏 F1 最大的判决阈值；并列时取最接近 0.5 者。"""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    scores = np.array([macro_f1(np.asarray(y).astype(int), (prob >= t).astype(int)) for t in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(grid[best[np.argmin(np.abs(grid[best] - 0.5))]])
