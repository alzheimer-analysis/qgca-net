"""为 AVEC2014 / MPSC 的视频批量调用 OpenFace 2.0 FeatureExtraction，输出每段视频的 CSV。

需要本机已安装 OpenFace 2.0（FeatureExtraction 可执行文件）。提取 AU 强度、头部姿态、视线与 2D 地标。
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="含 video 与 openface（输出 CSV 路径）两列的 manifest")
    ap.add_argument("--exe", required=True, help="FeatureExtraction 可执行文件路径")
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    for _, r in df.iterrows():
        out = Path(r["openface"])
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [args.exe, "-f", r["video"], "-aus", "-pose", "-gaze", "-2Dfp",
               "-out_dir", str(out.parent), "-of", out.stem]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
