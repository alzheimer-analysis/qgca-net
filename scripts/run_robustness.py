"""DAIC-WOZ 留出集上的缺失/损坏模态鲁棒性评估（QGCA-Net 及其无门控对照等）。

  python scripts/run_robustness.py --config configs/daic_woz.yaml --selection outputs/daic_woz/qgca/selection.json \
      --perturbed speech_noise_10db=data/processed/daic_woz_speech_noise_10db \
                  video_occlusion_30=data/processed/daic_woz_video_occlusion_30
"""
import argparse

import yaml

from _common import add_common_args, build_cfg, ensure_dir
from qgca.protocols.robustness import run_robustness
from qgca.utils import get_logger, load_json


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--robustness", default="configs/robustness.yaml")
    ap.add_argument("--selection", required=True, help="主实验输出的 selection.json（训练轮数与阈值，来自训练集 CV）")
    ap.add_argument("--perturbed", nargs="*", default=[], help="条件名=扰动后 processed 目录")
    args = ap.parse_args()
    cfg = build_cfg(args)
    out = args.out or f"{cfg.output_dir}/{cfg.dataset}/robustness_{cfg.model.name}"
    ensure_dir(out)
    get_logger("qgca", out + "/log.txt")
    with open(args.robustness, "r", encoding="utf-8") as f:
        rob = yaml.safe_load(f)
    dirs = dict(kv.split("=", 1) for kv in args.perturbed)
    run_robustness(cfg, rob, dirs, load_json(args.selection), out)


if __name__ == "__main__":
    main()
