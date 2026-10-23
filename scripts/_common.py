import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qgca.config import Cfg, apply_overrides, deep_update, load_config, update_cfg
from qgca.utils import ensure_dir, get_logger, load_json


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--config", required=True, help="数据集配置，如 configs/daic_woz.yaml")
    ap.add_argument("--extra", nargs="*", default=[], help="叠加在其上的其它 yaml（如 configs/baselines.yaml）")
    ap.add_argument("--set", nargs="*", default=[], help="dotted 覆盖，如 model.name=mult train.epochs=30")
    ap.add_argument("--hparams", default=None, help="run_hparam_search.py 输出的最优补丁 json")
    ap.add_argument("--out", default=None, help="输出目录")
    return ap


def build_cfg(args) -> Cfg:
    cfg = load_config(args.config, args.set)
    for e in args.extra:
        deep_update(cfg, load_config(e))
    if args.hparams:
        patch = load_json(args.hparams)
        patch = patch.get("best", patch).get("patch", patch.get("patch", patch))
        cfg = update_cfg(cfg, patch)
    cfg = update_cfg(cfg, {})            # 规范化为 Cfg
    return cfg


def out_dir_for(cfg, args, tag: str = "") -> str:
    base = args.out or os.path.join(cfg.output_dir, cfg.dataset, cfg.model.name + (f"_{tag}" if tag else ""))
    ensure_dir(base)
    return base
