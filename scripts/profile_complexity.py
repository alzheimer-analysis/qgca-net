"""复杂度（Table: Complexity Comparison, Per Interview, GPU）：参数量、GFLOPs、单次访谈推理延迟。

对每个方法构建模型，在一份真实访谈（取处理后数据集中的第一条）上统计:
  fusion params = 总参数量 - 三个预训练/单模态编码器参数量
  total params  = 全部参数量
  GFLOPs        = torch.utils.flop_counter 统计的前向浮点运算（注意力等算子按其规则计数）
  time (ms)     = 预热后多次前向的平均耗时（含同步）
"""
import argparse
import time

import torch

from _common import add_common_args, build_cfg
from qgca.data.dataset import Collator
from qgca.engine.trainer import build_dataset
from qgca.models.factory import build_model
from qgca.protocols import common as C
from qgca.utils import count_parameters, get_device, to_device


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--methods", nargs="+", default=["early_fusion", "mlan", "mult", "tamfn", "ttfnet", "qgca"])
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()
    base = build_cfg(args)
    device = get_device(base.device)
    index = C.load_index(base.data.processed_dir)
    for name in args.methods:
        from qgca.config import update_cfg
        cfg = update_cfg(base, {"model.name": name})
        model, cfg = build_model(cfg)
        model.to(device).eval()
        norm = C.fit_normalizer(C.select_entries(index, ["train"]), cfg)
        ds = build_dataset([C.select_entries(index, ["train"])[0]["path"]], norm, cfg, train=False)
        batch = to_device(Collator(cfg)([ds[0]]), device)
        enc = count_parameters(model.bank) if hasattr(model, "bank") else 0
        total = count_parameters(model)
        with torch.no_grad():
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                model(batch)
            gflops = fc.get_total_flops() / 1e9
            for _ in range(5):
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.repeats):
                model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / args.repeats * 1000
        print(f"{name:14s} fusion={(total - enc) / 1e6:.2f}M total={total / 1e6:.1f}M GFLOPs={gflops:.1f} time={ms:.1f}ms")


if __name__ == "__main__":
    main()
