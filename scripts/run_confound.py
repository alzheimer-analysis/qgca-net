"""质量指标混杂分析（Do the Quality Indicators Leak Label Information?）。

(1) DAIC-WOZ 训练集上每个候选指标与诊断标签的点二列相关（表 Association of Candidate Quality Indicators…）
(2) 仅用 q 的受试者级统计量训练逻辑回归，报告 DAIC-WOZ 留出集与 MPSC（5 折受试者分层 CV 的折外）AUC，
    行为类 q 与保真度 q 各一次。
四组对照配置（门控 + 行为 q、打乱 q 等）见 configs/controls.yaml，由 run_variants.py 运行。
"""
import argparse
import os

import numpy as np

from _common import add_common_args, build_cfg, ensure_dir
from qgca.data.splits import subject_stratified_kfold
from qgca.evaluation import confound as CF
from qgca.evaluation.metrics import auc
from qgca.evaluation.stats import hanley_mcneil_ci
from qgca.protocols import common as C
from qgca.utils import get_logger, save_json


def main():
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = build_cfg(args)
    out = args.out or f"{cfg.output_dir}/{cfg.dataset}/confound"
    ensure_dir(out)
    get_logger("qgca", out + "/log.txt")
    index = C.load_index(cfg.data.processed_dir)
    res = {}
    if cfg.dataset == "daic_woz":
        train = C.select_entries(index, ["train"])
        held = C.select_entries(index, cfg.protocol.eval_splits)
        res["correlations"] = CF.indicator_label_correlations(C.paths_of(train))
        norm = C.fit_normalizer(train, cfg)
        for kind in ("behavioral", "fidelity"):
            res[f"q_only_auc_{kind}"] = CF.q_only_auc(C.paths_of(train), C.paths_of(held), norm, cfg.quality, kind)
    else:                                                   # MPSC: 受试者分层 5 折折外 AUC
        tab = C.subject_table(index)
        subs = sorted(tab)
        folds = subject_stratified_kfold(subs, [f"{tab[s]['label']}_{tab[s]['age_group']}" for s in subs], 5, 0)
        for kind in ("behavioral", "fidelity"):
            ys, ps = [], []
            for tr_s, te_s in folds:
                tr, te = C.select_entries(index, subjects=tr_s), C.select_entries(index, subjects=te_s)
                norm = C.fit_normalizer(tr, cfg)
                Xtr, ytr = CF._q_stats(C.paths_of(tr), norm, cfg.quality, kind)
                Xte, yte = CF._q_stats(C.paths_of(te), norm, cfg.quality, kind)
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xtr, ytr)
                ys.append(yte)
                ps.append(clf.predict_proba(Xte)[:, 1])
            y, p = np.concatenate(ys), np.concatenate(ps)
            a = auc(y, p)
            lo, hi = hanley_mcneil_ci(a, int(y.sum()), int((1 - y).sum()))
            res[f"q_only_auc_{kind}"] = {"auc": a, "ci_low": lo, "ci_high": hi}
    save_json(res, os.path.join(out, "confound.json"))
    print(res)


if __name__ == "__main__":
    main()
