"""训练数据内的超参数搜索（所有方法同一预算：24 个配置，围绕已发表设置抽取）。

打分一律来自训练数据：DAIC-WOZ 训练集 5 折 CV / AVEC2014 dev 分区 / MPSC 内层 CV（对第一个外层折）。
输出 best 补丁 json，可通过 run_experiment.py --hparams 使用。
"""
import argparse
import os

import yaml

from _common import add_common_args, build_cfg, ensure_dir
from qgca.config import update_cfg
from qgca.data.splits import mpsc_nested_splits
from qgca.models.factory import prepare_model_cfg
from qgca.protocols import common as C
from qgca.protocols.daic import daic_folds
from qgca.utils import get_logger, save_json


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--space", default="configs/search_space.yaml")
    ap.add_argument("--method", default=None, help="qgca 或基线名；默认取 cfg.model.name")
    ap.add_argument("--epochs", type=int, default=None, help="搜索时的训练轮数（默认取 epochs_grid 的中位数）")
    args = ap.parse_args()
    cfg = prepare_model_cfg(build_cfg(args))
    out = args.out or os.path.join(cfg.output_dir, cfg.dataset, f"hparam_{cfg.model.name}")
    ensure_dir(out)
    get_logger("qgca", out + "/log.txt")
    with open(args.space, "r", encoding="utf-8") as f:
        space = yaml.safe_load(f)
    method = args.method or cfg.model.name
    sp = space["qgca"] if method in ("qgca", "qgca_speech") else {**space["baselines"]["common"], **space["baselines"].get(method, {})}
    cands = C.sample_candidates(sp, space["budget"], space["sampler_seed"])

    index = C.load_index(cfg.data.processed_dir)
    epochs = args.epochs or sorted(cfg.train.epochs_grid)[len(cfg.train.epochs_grid) // 2]
    if cfg.dataset == "daic_woz":
        train = C.select_entries(index, ["train"])
        folds, task, entries = daic_folds(train, cfg.protocol.cv_folds), "cls", train
    elif cfg.dataset == "avec2014":
        # AVEC2014: 以 train->dev 作为"单折"
        # AVEC2014: 在官方 dev 分区上打分（同一说话人可能出现在不同分区，故按分区而非受试者划分）
        train = C.select_entries(index, cfg.protocol.train_splits)
        dev = C.select_entries(index, [cfg.protocol.select_split])
        results = []
        for i, patch in enumerate(cands):
            c = prepare_model_cfg(update_cfg(cfg, patch))
            rows = C.train_and_predict(c, train, {"dev": dev}, epochs, C.seeds_of(c, selection=True))["dev"]
            results.append({"patch": patch, "score": float(C.metric_for_selection(rows, "reg")[0])})
        res = {"best": max(results, key=lambda r: r["score"]), "all": results}
        save_json(res, os.path.join(out, "search.json"))
        save_json(res["best"], os.path.join(out, "best.json"))
        return
    else:
        tab = C.subject_table(index)
        subs = sorted(tab)
        sp0 = mpsc_nested_splits(subs, [tab[s]["label"] for s in subs], [tab[s]["age_group"] for s in subs],
                                 cfg.protocol.outer_folds, 1, cfg.protocol.inner_folds)[0]
        entries, folds, task = C.select_entries(index, subjects=sp0["train"]), sp0["inner"], "cls"
    res = C.hyperparameter_search(cfg, entries, folds, task, cands, epochs)
    save_json(res, os.path.join(out, "search.json"))
    save_json(res["best"], os.path.join(out, "best.json"))


if __name__ == "__main__":
    main()
