"""鲁棒性扰动（DAIC-WOZ 留出集）。

* speech_noise_10db   : 对语音加白噪声，SNR 10 dB；重新提取 wav2vec 特征并重新计算语音保真度指标
* video_occlusion_30  : 随机遮挡 30% 的视频帧（帧置为无效、跟踪置信度计 0），重新计算跟踪置信度
* missing_*           : 整个模态缺失，在评估时通过 cfg.eval.drop_modalities 设置（不改动 record）

扰动对抑郁与健康受试者一视同仁，因此扰动下的收益不可能来自标签信息。
"""
import copy
from typing import Dict, Optional

import numpy as np

from .audio import Wav2VecExtractor, process_recording_audio


def speech_noise(record: Dict, w2v: Wav2VecExtractor, snr_db: float = 10.0, seed: int = 0) -> Dict:
    r = copy.deepcopy(record)
    segs = list(zip(r["utt_start"], r["utt_end"]))
    out = process_recording_audio(r["meta"]["audio_path"], segs, w2v, want_logmel=r.get("logmel") is not None,
                                  noise_snr_db=snr_db, rng=np.random.default_rng(seed))
    r["wav2vec"], r["audio_present"], r["logmel"] = out["wav2vec"], out["present"], out["logmel"]
    r["fid"]["A"] = out["fid"]
    r["behav"][:, 0], r["behav"][:, 1] = out["voiced"], out["energy"]
    return r


def video_occlusion(record: Dict, ratio: float = 0.30, seed: int = 0) -> Dict:
    r = copy.deepcopy(record)
    rng = np.random.default_rng(seed)
    for i in range(len(r["of_frames"])):
        occ = rng.random(len(r["of_valid"][i])) < ratio
        r["of_valid"][i] = np.asarray(r["of_valid"][i], bool) & ~occ
        r["of_conf"][i] = np.where(occ, 0.0, r["of_conf"][i]).astype(np.float32)
        if len(r["app_valid"][i]):
            r["app_valid"][i] = np.asarray(r["app_valid"][i], bool) & (rng.random(len(r["app_valid"][i])) >= ratio)
        r["fid"]["V"][i, 0] = float(r["of_conf"][i].mean()) if len(r["of_conf"][i]) else np.nan
    return r
