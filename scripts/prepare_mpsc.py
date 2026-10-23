"""MPSC 预处理（中文，青少年为主）。

输入 manifest CSV 列:
  record_id, subject_id, task(1..6), audio, video, openface,
  sync_audio_start, sync_video_start,   # 每个任务的逐试次时间戳文件给出的音频/视频起点（秒），用于音视频同步
  diagnosis(0/1, 心理医生诊断), phq9, age
流程: Whisper large-v3（语言固定为中文）转写并保留词级时间戳与对数概率 -> 在 >= 0.5 s 的静默处切分话语 ->
      按时间戳同步音视频 -> wav2vec / OpenFace / ViT 特征与保真度指标。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from qgca.preprocessing.audio import Wav2VecExtractor, process_recording_audio
from qgca.preprocessing.record import build_record, write_records
from qgca.preprocessing.text import WhisperASR, load_affect_lexicon, split_by_silence
from qgca.preprocessing.video import ViTExtractor, read_openface_csv, sample_video_frames, utterance_video_fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/processed/mpsc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--affect_lexicon", default=None)
    ap.add_argument("--no_logmel", action="store_true")
    ap.add_argument("--no_vit", action="store_true")
    ap.add_argument("--min_silence", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    asr = WhisperASR("large-v3", "zh")
    w2v = Wav2VecExtractor(device=args.device)
    vit = None if args.no_vit else ViTExtractor(device=args.device)
    lex = load_affect_lexicon(args.affect_lexicon)
    records = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        utts = [u for u in split_by_silence(asr.words(r["audio"]), args.min_silence) if u["text"]]
        if not utts:
            continue
        segs = [(u["start"], u["end"]) for u in utts]
        audio = process_recording_audio(r["audio"], segs, w2v, want_logmel=not args.no_logmel)
        # 音视频同步：视频时间 = 音频时间 + (视频起点 - 音频起点)
        offset = float(r["sync_video_start"]) - float(r["sync_audio_start"])
        vsegs = [(s + offset, e + offset) for s, e in segs]
        of = read_openface_csv(r["openface"])
        app = sample_video_frames(r["video"], of, vit=vit)
        video = utterance_video_fields(vsegs, of, app)
        age = float(r["age"])
        meta = {"subject_id": str(r["subject_id"]), "record_id": str(r["record_id"]), "dataset": "mpsc", "split": "all",
                "task": int(r["task"]), "label": int(r["diagnosis"]), "score": float(r["phq9"]), "age": age,
                "age_group": "13-19" if age <= 19 else "20-24",
                "meta": {"audio_path": r["audio"], "video_path": r["video"], "openface": r["openface"], "offset": offset}}
        records.append(build_record(meta, utts, audio, video, text_manual=False, affect_lexicon=lex))
    write_records(records, args.out)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
