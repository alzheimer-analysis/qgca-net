"""统计分析（论文 Metrics and Statistical Analysis）。

推断单位是受试者而非种子：先对 5 个种子的预测按受试者取平均。
  * 宏 F1 / RMSE 的 95% CI: 按受试者的分层 bootstrap（10 000 次）
  * AUC 的 CI: Hanley–McNeil
  * QGCA-Net 与各竞争者的差异: 配对分层 bootstrap（宏 F1、RMSE）与 DeLong 检验（AUC）
  * 每个语料每个指标的 8 个比较的 p 值做 Holm–Bonferroni 校正
  * 合并分析: DAIC-WOZ + MPSC，bootstrap 按语料分层
CI 不做多重校正；p 值经 Holm 校正。
"""
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# ----------------------------------------------------------------------------- bootstrap
def _strata_indices(strata: np.ndarray) -> List[np.ndarray]:
    return [np.flatnonzero(strata == s) for s in np.unique(strata)]


def bootstrap_indices(strata: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """分层有放回重采样的下标矩阵 [n_boot, N]（每层内部独立重采样，保持层大小）。"""
    groups = _strata_indices(np.asarray(strata))
    N = len(strata)
    out = np.empty((n_boot, N), dtype=np.int64)
    for b in range(n_boot):
        out[b] = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
    return out


def bootstrap_ci(metric: Callable[[np.ndarray], float], n: int, strata: np.ndarray, n_boot: int = 10000,
                 seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """metric(idx) -> 标量，idx 为参与计算的受试者下标。返回 (点估计, 下界, 上界)。"""
    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(strata, n_boot, rng)
    vals = np.array([metric(i) for i in idx])
    vals = vals[np.isfinite(vals)]
    return float(metric(np.arange(n))), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(metric: Callable[[np.ndarray, np.ndarray], float], n: int, strata: np.ndarray,
                          n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """metric(idx, which) 其中 which∈{0,1} 选择模型 A/B。返回差值 A-B 的点估计、CI、双侧 p 值与 CI 半宽。

    对同一批重采样受试者同时计算 A 与 B（配对）。p = 2·min(P(Δ*≤0), P(Δ*≥0))。
    """
    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(strata, n_boot, rng)
    d = np.array([metric(i, 0) - metric(i, 1) for i in idx])
    d = d[np.isfinite(d)]
    point = metric(np.arange(n), 0) - metric(np.arange(n), 1)
    lo, hi = np.quantile(d, alpha / 2), np.quantile(d, 1 - alpha / 2)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(point), "ci": (float(lo), float(hi)), "half_width": float((hi - lo) / 2),
            "p": float(min(1.0, max(p, 1.0 / (len(d) + 1))))}


# ----------------------------------------------------------------------------- AUC
def hanley_mcneil_ci(auc_value: float, n_pos: int, n_neg: int, alpha: float = 0.05):
    q1 = auc_value / (2 - auc_value)
    q2 = 2 * auc_value ** 2 / (1 + auc_value)
    se = np.sqrt((auc_value * (1 - auc_value) + (n_pos - 1) * (q1 - auc_value ** 2)
                  + (n_neg - 1) * (q2 - auc_value ** 2)) / (n_pos * n_neg))
    z = sps.norm.ppf(1 - alpha / 2)
    return float(max(0.0, auc_value - z * se)), float(min(1.0, auc_value + z * se))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(preds: np.ndarray, m: int):
    """preds:[K,N]，前 m 个为正类样本。返回各分类器 AUC 及其协方差（Sun & Xu, 2014）。"""
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx, ty, tz = (np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n)))
    for r in range(k):
        tx[r], ty[r], tz[r] = _midrank(pos[r]), _midrank(neg[r]), _midrank(preds[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(s)


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Dict[str, float]:
    """配对 DeLong 检验（双侧）：H0 AUC1 = AUC2。"""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, s = _fast_delong(preds, m)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff), "z": float(z),
            "p": float(2 * sps.norm.sf(abs(z)))}


# ----------------------------------------------------------------------------- 多重比较
def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """Holm–Bonferroni 校正后的 p 值（保持输入顺序）。"""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ----------------------------------------------------------------------------- 便捷封装
def macro_f1_ci(y: np.ndarray, prob: np.ndarray, threshold: float, n_boot=10000, seed=0):
    from .metrics import macro_f1
    pred = (prob >= threshold).astype(int)
    return bootstrap_ci(lambda i: macro_f1(y[i], pred[i]), len(y), y, n_boot, seed)


def rmse_ci(y: np.ndarray, p: np.ndarray, strata: np.ndarray = None, n_boot=10000, seed=0):
    return bootstrap_ci(lambda i: float(np.sqrt(np.mean((y[i] - p[i]) ** 2))), len(y),
                        np.zeros(len(y)) if strata is None else strata, n_boot, seed)


def compare_models(y: np.ndarray, probs: Dict[str, np.ndarray], thresholds: Dict[str, float], ref: str,
                   n_boot: int = 10000, seed: int = 0) -> Dict[str, Dict]:
    """参考模型 ref（QGCA-Net）与每个竞争者的宏 F1 配对 bootstrap 差值与 DeLong AUC 检验，并做 Holm 校正。"""
    from .metrics import macro_f1
    y = np.asarray(y).astype(int)
    pr = {k: (np.asarray(v) >= thresholds[k]).astype(int) for k, v in probs.items()}
    res, f1_p, auc_p, keys = {}, [], [], []
    for k in probs:
        if k == ref:
            continue
        fn = lambda i, w, k=k: macro_f1(y[i], pr[ref][i] if w == 0 else pr[k][i])
        d = paired_bootstrap_diff(fn, len(y), y, n_boot, seed)
        dl = delong_test(y, probs[ref], probs[k])
        res[k] = {"f1_diff": d["diff"], "f1_ci": d["ci"], "f1_half_width": d["half_width"], "f1_p_raw": d["p"],
                  "auc_diff": dl["diff"], "auc_p_raw": dl["p"]}
        keys.append(k)
        f1_p.append(d["p"])
        auc_p.append(dl["p"])
    for k, a, b in zip(keys, holm_adjust(f1_p), holm_adjust(auc_p)):
        res[k]["f1_p_holm"], res[k]["auc_p_holm"] = a, b
    return res


def compare_rmse(y: np.ndarray, preds: Dict[str, np.ndarray], ref: str, subject_ids: np.ndarray,
                 n_boot: int = 10000, seed: int = 0) -> Dict[str, Dict]:
    """AVEC2014：按受试者（而非录音）重采样的 RMSE 配对差值。subject_ids 与 y 逐录音对应。"""
    uniq = np.unique(subject_ids)
    members = [np.flatnonzero(subject_ids == s) for s in uniq]

    def rm(idx_subj, which_pred):
        rec = np.concatenate([members[j] for j in idx_subj])
        return float(np.sqrt(np.mean((y[rec] - which_pred[rec]) ** 2)))

    res, ps, keys = {}, [], []
    for k, p in preds.items():
        if k == ref:
            continue
        d = paired_bootstrap_diff(lambda i, w, k=k: rm(i, preds[ref] if w == 0 else preds[k]),
                                  len(uniq), np.zeros(len(uniq)), n_boot, seed)
        res[k] = {"rmse_diff": d["diff"], "ci": d["ci"], "half_width": d["half_width"], "p_raw": d["p"]}
        ps.append(d["p"])
        keys.append(k)
    for k, a in zip(keys, holm_adjust(ps)):
        res[k]["p_holm"] = a
    return res


def pooled_compare(per_corpus: Dict[str, Dict], ref: str, other: str, n_boot: int = 10000, seed: int = 0) -> Dict:
    """合并分析（如 DAIC-WOZ + MPSC）：bootstrap 按语料（及标签）分层。
    per_corpus[corpus] = {"y": ..., "pred": {model: 0/1 预测}}"""
    from .metrics import macro_f1
    ys, pa, pb, st = [], [], [], []
    for c, d in per_corpus.items():
        ys.append(np.asarray(d["y"]).astype(int))
        pa.append(np.asarray(d["pred"][ref]))
        pb.append(np.asarray(d["pred"][other]))
        st.append(np.array([f"{c}_{v}" for v in d["y"]]))
    y, a, b, strata = map(np.concatenate, (ys, pa, pb, st))
    return paired_bootstrap_diff(lambda i, w: macro_f1(y[i], a[i] if w == 0 else b[i]), len(y), strata, n_boot, seed)
