"""生成扰动后的 DAIC-WOZ 留出集 record（语音加噪 / 视频遮挡），写入独立目录并附 index.json。

  python scripts/make_perturbed_records.py --processed data/processed/daic_woz --kind speech_noise --snr_db 10 \
         --out data/processed/daic_woz_speech_noise_10db
  python scripts/make_perturbed_records.py --processed data/processed/daic_woz --kind video_occlusion --ratio 0.3 \
         --out data/processed/daic_woz_video_occlusion_30
只处理 dev/test 分区（留出集）。同一扰动对抑郁和健康受试者完全相同。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from qgca.data.schema import load_record
from qgca.preprocessing.audio import Wav2VecExtractor
from qgca.preprocessing.perturb import speech_noise, video_occlusion
from qgca.preprocessing.record import write_records
from qgca.utils import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--kind", choices=["speech_noise", "video_occlusion"], required=True)
    ap.add_argument("--snr_db", type=float, default=10.0)
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--splits", nargs="*", default=["dev", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = [e for e in load_json(os.path.join(args.processed, "index.json")) if e["split"] in args.splits]
    w2v = Wav2VecExtractor(device=args.device) if args.kind == "speech_noise" else None
    out = []
    for e in tqdm(index):
        r = load_record(os.path.join(args.processed, e["path"]))
        if args.kind == "speech_noise":
            r = speech_noise(r, w2v, args.snr_db, args.seed)
        else:
            r = video_occlusion(r, args.ratio, args.seed)
        out.append(r)
    write_records(out, args.out)
    print(f"wrote {len(out)} perturbed records to {args.out}")


if __name__ == "__main__":
    main()
