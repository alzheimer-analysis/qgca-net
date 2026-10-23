"""批量运行消融（configs/ablation.yaml）或混杂对照（configs/controls.yaml）中的全部变体。

每个变体 = 基础配置 + dotted 补丁。变体配置在评估前固定，其结果不用于修改所提模型。
示例:
  python scripts/run_variants.py --config configs/daic_woz.yaml --variants configs/ablation.yaml
  python scripts/run_variants.py --config configs/mpsc.yaml --variants configs/controls.yaml
"""
import argparse
import os

import yaml

from _common import add_common_args, build_cfg, ensure_dir
from qgca.config import update_cfg
from qgca.protocols.avec import run_avec
from qgca.protocols.daic import run_daic
from qgca.protocols.mpsc import run_mpsc
from qgca.utils import get_logger, save_json


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--variants", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    base = build_cfg(args)
    with open(args.variants, "r", encoding="utf-8") as f:
        variants = yaml.safe_load(f)["variants"]
    root = args.out or os.path.join(base.output_dir, base.dataset, "variants_" + os.path.basename(args.variants).split(".")[0])
    ensure_dir(root)
    summary = {}
    for name, patch in variants.items():
        if args.only and name not in args.only:
            continue
        cfg = update_cfg(base, patch)
        out = os.path.join(root, name)
        ensure_dir(out)
        get_logger("qgca", out + "/log.txt")
        save_json({"patch": patch, "config": cfg.to_dict()}, out + "/config.json")
        if cfg.dataset == "daic_woz":
            summary[name] = run_daic(cfg, out)
        elif cfg.dataset == "avec2014":
            summary[name] = run_avec(cfg, out)
        else:
            summary[name] = run_mpsc(cfg, out)
    save_json(summary, os.path.join(root, "summary.json"))


if __name__ == "__main__":
    main()
