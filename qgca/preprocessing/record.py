"""把文本/语音/视频的话语级产物装配成统一 record（字段见 qgca/data/schema.py）并写出索引。"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.schema import save_record
from ..utils import ensure_dir, save_json
from .text import behavior_text


def build_record(meta: Dict, utts: List[Dict], audio: Dict, video: Dict, text_manual: bool,
                 affect_lexicon: set) -> Dict:
    """
    meta  : subject_id, record_id, dataset, split, task, label, score, age, age_group, 以及各类路径
    utts  : [{start, end, text, asr_logprob(可无)}]
    audio : process_recording_audio 的输出
    video : utterance_video_fields 的输出
    """
    n = len(utts)
    texts = [u.get("text", "") for u in utts]
    text_present = np.array([len(t.strip()) > 0 for t in texts], bool)
    asr = np.array([np.nan if text_manual else u.get("asr_logprob", np.nan) for u in utts], np.float64)
    tok_cnt, aff = np.zeros(n), np.zeros(n)
    for i, t in enumerate(texts):
        tok_cnt[i], aff[i] = behavior_text(t, affect_lexicon)
    behav = np.stack([audio["voiced"], audio["energy"], tok_cnt, aff, video["head_pose"], video["gaze"]], 1)
    rec = dict(meta)
    rec.update({
        "utt_start": np.array([u["start"] for u in utts], np.float64),
        "utt_end": np.array([u["end"] for u in utts], np.float64),
        "text": texts, "text_present": text_present, "text_manual": bool(text_manual), "asr_logprob": asr,
        "wav2vec": audio["wav2vec"], "logmel": audio["logmel"], "audio_present": audio["present"],
        "of_frames": video["of_frames"], "of_valid": video["of_valid"], "of_conf": video["of_conf"],
        "app_frames": video["app_frames"], "app_valid": video["app_valid"],
        "fid": {"T": asr[:, None], "A": audio["fid"], "V": video["fid"]},
        "behav": behav,
    })
    return rec


def write_records(records: Sequence[Dict], out_dir: str) -> None:
    """逐条保存 .pt 并写 index.json（供协议读取，避免反复加载大文件）。"""
    ensure_dir(out_dir)
    index = []
    for r in records:
        rel = f"{r['record_id']}.pt"
        save_record(r, os.path.join(out_dir, rel))
        index.append({"record_id": r["record_id"], "subject_id": r["subject_id"], "path": rel, "split": r["split"],
                      "label": int(r["label"]), "score": None if r.get("score") is None or not np.isfinite(r["score"])
                      else float(r["score"]), "age": r.get("age"), "age_group": r.get("age_group"),
                      "task": r.get("task"), "dataset": r["dataset"]})
    save_json(index, os.path.join(out_dir, "index.json"))
