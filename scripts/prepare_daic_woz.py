"""DAIC-WOZ 预处理：人工转写 + 音频 + 发布的逐帧视觉特征 -> 话语级对齐 record。

目录约定（官方发布）:
  <root>/<pid>_P/<pid>_AUDIO.wav, <pid>_TRANSCRIPT.csv, <pid>_CLNF_{AUs,pose,gaze,hog,features3D}.txt
  <labels_dir>/{train_split_Depression_AVEC2017.csv, dev_split_Depression_AVEC2017.csv, full_test_split.csv}
话语边界取自转写时间戳，只保留受访者（Participant）的话语。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from qgca.preprocessing.audio import Wav2VecExtractor, process_recording_audio
from qgca.preprocessing.record import build_record, write_records
from qgca.preprocessing.text import load_affect_lexicon, parse_daic_transcript
from qgca.preprocessing.video import read_daic_visual, utterance_video_fields


def read_labels(labels_dir: str) -> pd.DataFrame:
    parts = []
    for split, fn, bin_col, score_col in [("train", "train_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
                                          ("dev", "dev_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
                                          ("test", "full_test_split.csv", "PHQ_Binary", "PHQ_Score")]:
        df = pd.read_csv(os.path.join(labels_dir, fn))
        parts.append(pd.DataFrame({"pid": df["Participant_ID"].astype(str), "split": split,
                                   "label": df[bin_col].astype(int), "score": df[score_col].astype(float)}))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out", default="data/processed/daic_woz")
    ap.add_argument("--affect_lexicon", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w2v", default="facebook/wav2vec2-base")
    ap.add_argument("--no_logmel", action="store_true")
    args = ap.parse_args()

    lex = load_affect_lexicon(args.affect_lexicon)
    w2v = Wav2VecExtractor(args.w2v, args.device)
    records = []
    for _, row in tqdm(read_labels(args.labels_dir).iterrows(), total=len(read_labels(args.labels_dir))):
        pid, sdir = row["pid"], os.path.join(args.root, f"{row['pid']}_P")
        utts = [u for u in parse_daic_transcript(os.path.join(sdir, f"{pid}_TRANSCRIPT.csv")) if u["text"]]
        if not utts:
            continue
        segs = [(u["start"], u["end"]) for u in utts]
        audio = process_recording_audio(os.path.join(sdir, f"{pid}_AUDIO.wav"), segs, w2v, want_logmel=not args.no_logmel)
        vis = read_daic_visual(sdir, pid)
        app = {"t": vis["app_t"], "app": vis["app"], "ok": vis["app_ok"]}
        video = utterance_video_fields(segs, vis, app)
        meta = {"subject_id": pid, "record_id": pid, "dataset": "daic_woz", "split": row["split"], "task": "interview",
                "label": int(row["label"]), "score": float(row["score"]), "age": None, "age_group": None,
                "meta": {"audio_path": os.path.join(sdir, f"{pid}_AUDIO.wav"), "session_dir": sdir}}
        records.append(build_record(meta, utts, audio, video, text_manual=True, affect_lexicon=lex))
    write_records(records, args.out)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
