"""AVEC2014 预处理（Northwind 与 Freeform 任务，仅语音+视频，文本分支整体被掩码）。

输入 manifest CSV 列: record_id, subject_id, task, split(train|dev|test), audio, video, openface, bdi
  * openface: 对该录音运行 OpenFace 2.0 FeatureExtraction（-aus -pose -gaze -2Dfp）得到的 CSV
话语（语音片段）由双门限 VAD 得到。风险标签 BDI-II >= 14。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from qgca.preprocessing.audio import Wav2VecExtractor, dual_threshold_vad, load_audio, process_recording_audio, vad_segments
from qgca.preprocessing.record import build_record, write_records
from qgca.preprocessing.text import load_affect_lexicon
from qgca.preprocessing.video import ViTExtractor, read_openface_csv, sample_video_frames, utterance_video_fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/processed/avec2014")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_logmel", action="store_true")
    ap.add_argument("--no_vit", action="store_true", help="仅 OpenFace 特征（跨语料实验使用时可省略 ViT）")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    w2v = Wav2VecExtractor(device=args.device)
    vit = None if args.no_vit else ViTExtractor(device=args.device)
    lex = load_affect_lexicon(None)
    records = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        wav, _, _ = load_audio(r["audio"])
        segs = vad_segments(dual_threshold_vad(wav))
        if not segs:
            continue
        utts = [{"start": s, "end": e, "text": ""} for s, e in segs]            # 无转写
        audio = process_recording_audio(r["audio"], segs, w2v, want_logmel=not args.no_logmel)
        of = read_openface_csv(r["openface"])
        app = sample_video_frames(r["video"], of, vit=vit)
        video = utterance_video_fields(segs, of, app)
        bdi = float(r["bdi"])
        meta = {"subject_id": str(r["subject_id"]), "record_id": str(r["record_id"]), "dataset": "avec2014",
                "split": r["split"], "task": r["task"], "label": int(bdi >= 14), "score": bdi, "age": None,
                "age_group": None, "meta": {"audio_path": r["audio"], "video_path": r["video"], "openface": r["openface"]}}
        records.append(build_record(meta, utts, audio, video, text_manual=False, affect_lexicon=lex))
    write_records(records, args.out)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
