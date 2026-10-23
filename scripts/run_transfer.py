"""迁移实验（configs/transfer.yaml）。

跨语料:
  python scripts/run_transfer.py cross_corpus --direction avec_to_daic --dsh 1 --dal 1 \
      --daic_cfg configs/daic_woz.yaml --avec_cfg configs/avec2014.yaml \
      --source_epochs 30 --finetune_epochs 15 --threshold 0.5
跨语言/青少年 (DAIC-WOZ -> MPSC):
  python scripts/run_transfer.py cross_lingual --method qgca --daic_cfg configs/daic_woz.yaml --mpsc_cfg configs/mpsc.yaml \
      --source_epochs 30 --finetune_epochs 15 --threshold 0.5
源域训练轮数与阈值应取自源域训练数据内的交叉验证（对应协议的 selection.json）。
"""
import argparse
import os

from _common import ensure_dir
from qgca.config import load_config
from qgca.protocols.transfer import run_cross_corpus, run_cross_lingual
from qgca.utils import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cross_corpus", "cross_lingual"])
    ap.add_argument("--daic_cfg", default="configs/daic_woz.yaml")
    ap.add_argument("--avec_cfg", default="configs/avec2014.yaml")
    ap.add_argument("--mpsc_cfg", default="configs/mpsc.yaml")
    ap.add_argument("--direction", choices=["avec_to_daic", "daic_to_avec"], default="avec_to_daic")
    ap.add_argument("--method", default="qgca")
    ap.add_argument("--dsh", type=int, default=1)
    ap.add_argument("--dal", type=int, default=1)
    ap.add_argument("--source_epochs", type=int, default=30)
    ap.add_argument("--finetune_epochs", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--only_repeat", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"outputs/transfer/{args.mode}_{args.direction if args.mode == 'cross_corpus' else args.method}_dsh{args.dsh}_dal{args.dal}"
    ensure_dir(out)
    get_logger("qgca", out + "/log.txt")
    setting = {"source": args.source_epochs, "source_epochs": args.source_epochs, "finetune": args.finetune_epochs,
               "finetune_epochs": args.finetune_epochs, "threshold": args.threshold}
    if args.mode == "cross_corpus":
        cfgs = {"daic_woz": load_config(args.daic_cfg, args.set), "avec2014": load_config(args.avec_cfg, args.set)}
        run_cross_corpus(cfgs, args.direction, bool(args.dsh), bool(args.dal), setting, out)
    else:
        patch = {"model.name": args.method}
        run_cross_lingual(load_config(args.daic_cfg, args.set), load_config(args.mpsc_cfg, args.set), patch,
                          bool(args.dsh), bool(args.dal), setting, out, args.only_repeat)


if __name__ == "__main__":
    main()
