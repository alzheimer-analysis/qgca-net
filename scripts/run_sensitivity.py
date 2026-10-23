"""超参数敏感性（configs/sensitivity.yaml）：在 DAIC-WOZ 训练集 5 折交叉验证上扫描四个超参数
（模态 Dropout、熵正则权重、掩码阈值、注意力头数）。这些曲线用于选择，不涉及留出集。"""
import argparse
import os

import yaml

from _common import add_common_args, build_cfg, ensure_dir
from qgca.config import update_cfg
from qgca.models.factory import prepare_model_cfg
from qgca.protocols import common as C
from qgca.protocols.daic import daic_folds
from qgca.utils import get_logger, save_json


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--sweeps", default="configs/sensitivity.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    base = build_cfg(args)
    assert base.dataset == "daic_woz", "敏感性分析在 DAIC-WOZ 训练集 CV 上进行"
    out = args.out or f"{base.output_dir}/daic_woz/sensitivity"
    ensure_dir(out)
    get_logger("qgca", out + "/log.txt")
    with open(args.sweeps, "r", encoding="utf-8") as f:
        sweeps = yaml.safe_load(f)["sweeps"]
    train = C.select_entries(C.load_index(base.data.processed_dir), ["train"])
    folds = daic_folds(train, base.protocol.cv_folds)
    epochs = args.epochs or sorted(base.train.epochs_grid)[len(base.train.epochs_grid) // 2]
    res = {}
    for name, sw in sweeps.items():
        res[name] = []
        for v in sw["values"]:
            cfg = prepare_model_cfg(update_cfg(base, {sw["key"]: v}))
            score = C.cv_score(cfg, train, folds, "cls", epochs, C.seeds_of(cfg, selection=True))
            res[name].append({"value": v, "cv_macro_f1": float(score)})
            print(name, v, score)
    save_json(res, os.path.join(out, "sensitivity.json"))


if __name__ == "__main__":
    main()
