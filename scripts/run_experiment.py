"""主实验入口：按 cfg.dataset 调用对应协议（DAIC-WOZ / AVEC2014 / MPSC）。

示例:
  python scripts/run_experiment.py --config configs/daic_woz.yaml --out outputs/daic_woz/qgca
  python scripts/run_experiment.py --config configs/daic_woz.yaml --extra configs/baselines.yaml --set model.name=mult
"""
import argparse

from _common import add_common_args, build_cfg, out_dir_for, load_json
from qgca.protocols.avec import run_avec
from qgca.protocols.daic import run_daic
from qgca.protocols.mpsc import run_mpsc
from qgca.utils import get_logger, save_json


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--selection", default=None, help="已有的 selection.json（跳过轮数/阈值选择，仅用于复现）")
    ap.add_argument("--only_repeat", type=int, default=None, help="MPSC: 只跑某一次外层重复（便于分批）")
    args = ap.parse_args()
    cfg = build_cfg(args)
    out = out_dir_for(cfg, args)
    get_logger("qgca", out + "/log.txt")
    save_json(cfg.to_dict(), out + "/config.json")
    sel = load_json(args.selection) if args.selection else None
    if cfg.dataset == "daic_woz":
        run_daic(cfg, out, sel)
    elif cfg.dataset == "avec2014":
        run_avec(cfg, out, sel)
    elif cfg.dataset == "mpsc":
        run_mpsc(cfg, out, args.only_repeat)
    else:
        raise ValueError(cfg.dataset)


if __name__ == "__main__":
    main()
