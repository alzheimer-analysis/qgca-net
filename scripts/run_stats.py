"""被试级统计比较（Statistical Comparison）与校准指标。

读取各方法的主实验输出目录（predictions_*.json、selection.json / metrics.json），以 --ref 指定的方法
（QGCA-Net）为参照，对每个竞争者:
  * 宏 F1 差值: 配对分层 bootstrap（10 000 次）；AUC 差值: DeLong 检验
  * 每个语料每个指标的比较做 Holm–Bonferroni 校正
  * 提供 --runs_mpsc 时再做 DAIC-WOZ + MPSC 的合并分析（bootstrap 按语料分层）
AVEC2014 回归（--corpus avec）按受试者重采样，比较 RMSE 差值。
示例:
  python scripts/run_stats.py --corpus daic --ref qgca --runs qgca=outputs/daic_woz/qgca mult=outputs/daic_woz/mult ...
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qgca.engine.trainer import aggregate_subjects
from qgca.evaluation import metrics as M
from qgca.evaluation import stats as S
from qgca.utils import load_json, save_json

FILES = {"daic": ("predictions_heldout.json", "selection.json", "threshold"),
         "mpsc": ("predictions_merged.json", "metrics.json", "mean_threshold"),
         "avec": ("predictions_test.json", None, None)}


def load_run(run_dir: str, corpus: str):
    pf, tf, key = FILES[corpus]
    rows = load_json(os.path.join(run_dir, pf))
    thr = load_json(os.path.join(run_dir, tf))[key] if tf else 0.5
    return rows, thr


def align(runs, corpus):
    """按受试者（avec: 按录音）对齐各方法的预测。"""
    tables = {}
    for name, d in runs.items():
        rows, thr = load_run(d, corpus)
        subj = rows if corpus == "avec" else aggregate_subjects(rows)
        keyf = "record_id" if corpus == "avec" else "subject_id"
        tables[name] = ({r[keyf]: r for r in subj}, thr, rows)
    ids = sorted(set.intersection(*[set(t[0]) for t in tables.values()]))
    return ids, tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["daic", "mpsc", "avec"], required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="方法名=输出目录")
    ap.add_argument("--runs_mpsc", nargs="*", default=None, help="合并分析所需的 MPSC 输出目录: 方法名=目录")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", default="outputs/stats.json")
    args = ap.parse_args()
    runs = dict(kv.split("=", 1) for kv in args.runs)
    ids, tab = align(runs, args.corpus)
    res = {"n": len(ids)}

    if args.corpus == "avec":
        y = np.array([tab[args.ref][0][i]["score"] for i in ids])
        preds = {k: np.array([t[0][i]["reg"] for i in ids]) for k, t in tab.items()}
        sid = np.array([tab[args.ref][0][i]["subject_id"] for i in ids])
        res["compare_rmse"] = S.compare_rmse(y, preds, args.ref, sid, args.n_boot)
    else:
        y = np.array([tab[args.ref][0][i]["y"] for i in ids]).astype(int)
        probs = {k: np.array([t[0][i]["prob"] for i in ids]) for k, t in tab.items()}
        thr = {k: t[1] for k, t in tab.items()}
        res["compare"] = S.compare_models(y, probs, thr, args.ref, args.n_boot)
        pr = probs[args.ref]
        res["calibration"] = {"brier": M.brier(y, pr), "ece": M.ece(y, pr, n_bins=6, equal_count=True)}
        if args.runs_mpsc:
            runs2 = dict(kv.split("=", 1) for kv in args.runs_mpsc)
            ids2, tab2 = align(runs2, "mpsc")
            y2 = np.array([tab2[args.ref][0][i]["y"] for i in ids2]).astype(int)
            pooled = {}
            for k in probs:
                if k == args.ref or k not in tab2:
                    continue
                per = {"daic": {"y": y, "pred": {args.ref: (probs[args.ref] >= thr[args.ref]).astype(int),
                                                  k: (probs[k] >= thr[k]).astype(int)}},
                       "mpsc": {"y": y2, "pred": {a: (np.array([tab2[a][0][i]["prob"] for i in ids2]) >= tab2[a][1]).astype(int)
                                                  for a in (args.ref, k)}}}
                pooled[k] = S.pooled_compare(per, args.ref, k, args.n_boot)
            ps = S.holm_adjust([v["p"] for v in pooled.values()])
            for (k, v), p in zip(pooled.items(), ps):
                v["p_holm"] = p
            res["pooled_daic_mpsc"] = pooled
    save_json(res, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
