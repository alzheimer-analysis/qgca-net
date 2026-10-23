"""语音预处理：重采样到 16 kHz、预加重、双门限 VAD、话语切分、wav2vec 2.0 特征与语音保真度指标。

语音保真度（越大越好）:
  noise_floor_snr      同一话语内"语音帧功率 / 非语音帧功率"(dB)，不依赖沉默时长
  clipping_ratio_inv   1 - 接近满量程(0.1% 以内)的采样点比例（在原始采样率波形上计算）
  effective_bandwidth  语音帧的 95% 谱滚降 (Hz) 的均值
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SR = 16000
FRAME = int(0.025 * SR)      # 25 ms
HOP = int(0.010 * SR)        # 10 ms（log-Mel 分支: 40 个频带、25 ms 窗、10 ms 帧移）
N_MELS = 40
LOGMEL_FRAMES = 300          # 每个话语的对数梅尔谱定长（截断/补零）


# ---------------------------------------------------------------------------- 读取与基本处理
def load_audio(path: str, sr: int = SR) -> Tuple[np.ndarray, np.ndarray, int]:
    """返回 (16 kHz 单声道波形, 原始采样率单声道波形, 原始采样率)。波形幅度归一到 [-1, 1]。"""
    import librosa
    import soundfile as sf
    native, native_sr = sf.read(path, dtype="float32", always_2d=True)
    native = native.mean(1)
    wav = native if native_sr == sr else librosa.resample(native, orig_sr=native_sr, target_sr=sr)
    return wav.astype(np.float32), native, int(native_sr)


def pre_emphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - 0.97 x[n-1]"""
    return np.append(x[:1], x[1:] - coef * x[:-1]).astype(np.float32)


def frame_signal(x: np.ndarray, frame: int = FRAME, hop: int = HOP) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


# ---------------------------------------------------------------------------- 双门限 VAD
def dual_threshold_vad(x: np.ndarray, sr: int = SR, hi_frac: float = 0.5, lo_frac: float = 0.2,
                       zcr_thr: float = 0.25) -> np.ndarray:
    """基于短时能量与过零率的双门限端点检测，返回逐帧语音标记（10 ms 帧移）。

    高能量阈值判定语音核心；与核心相邻、能量高于低阈值或过零率高于 zcr_thr（清音）的帧并入语音。
    阈值由 10% / 95% 能量分位数得到的噪声底与峰值自适应确定。
    """
    fr = frame_signal(pre_emphasis(x), FRAME, HOP)
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
    zcr = (np.abs(np.diff(np.sign(fr), axis=1)) > 0).mean(1)
    noise, peak = np.percentile(e, 10), np.percentile(e, 95)
    t_hi, t_lo = noise + hi_frac * (peak - noise), noise + lo_frac * (peak - noise)
    core = e > t_hi
    ext = (e > t_lo) | (zcr > zcr_thr)
    speech = core.copy()
    # 向两侧扩展：与核心连续相邻的 ext 帧并入
    for i in range(1, len(speech)):
        if not speech[i] and speech[i - 1] and ext[i]:
            speech[i] = True
    for i in range(len(speech) - 2, -1, -1):
        if not speech[i] and speech[i + 1] and ext[i]:
            speech[i] = True
    return speech


def vad_segments(speech: np.ndarray, min_gap: float = 0.3, min_dur: float = 0.3) -> List[Tuple[float, float]]:
    """由逐帧标记得到 [start, end]（秒）片段：合并间隔 < min_gap 的片段，丢弃短于 min_dur 的片段。"""
    segs, start = [], None
    for i, s in enumerate(speech):
        if s and start is None:
            start = i
        if (not s) and start is not None:
            segs.append([start * HOP / SR, i * HOP / SR])
            start = None
    if start is not None:
        segs.append([start * HOP / SR, len(speech) * HOP / SR])
    merged: List[List[float]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_dur]


# ---------------------------------------------------------------------------- 保真度与行为指标
def utterance_fidelity(seg16: np.ndarray, seg_native: np.ndarray, speech_frames: np.ndarray) -> np.ndarray:
    """返回 [snr_db, clipping_inv, bandwidth_hz]；无法计算的项为 NaN。"""
    import librosa
    fr = frame_signal(seg16, FRAME, HOP)
    power = (fr ** 2).mean(1) + 1e-12
    k = min(len(power), len(speech_frames))
    power, sp = power[:k], speech_frames[:k].astype(bool)
    snr = np.nan
    if sp.any():
        noise = power[~sp]
        if len(noise) == 0:                                   # 话语内没有非语音帧：取能量最低的 10% 帧作噪声底
            noise = np.sort(power)[: max(1, int(0.1 * len(power)))]
        snr = 10 * np.log10(power[sp].mean() / noise.mean())
    clip = 1.0 - float((np.abs(seg_native) >= 0.999).mean()) if len(seg_native) else np.nan
    bw = np.nan
    if sp.any() and len(seg16) >= 512:
        roll = librosa.feature.spectral_rolloff(y=seg16, sr=SR, n_fft=512, hop_length=HOP, roll_percent=0.95)[0]
        k2 = min(len(roll), len(sp))
        m2 = sp[:k2]
        bw = float(roll[:k2][m2].mean()) if m2.any() else np.nan
    return np.array([snr, clip, bw], dtype=np.float64)


def utterance_behavior_audio(seg16: np.ndarray, speech_frames: np.ndarray) -> Tuple[float, float]:
    """行为类指标（仅混杂分析）：发声比例与语音能量(dB)。"""
    fr = frame_signal(seg16, FRAME, HOP)
    k = min(len(fr), len(speech_frames))
    sp = speech_frames[:k].astype(bool)
    if k == 0:
        return np.nan, np.nan
    voiced = float(sp.mean())
    energy = float(10 * np.log10((fr[:k][sp] ** 2).mean() + 1e-12)) if sp.any() else np.nan
    return voiced, energy


def log_mel(seg16: np.ndarray) -> np.ndarray:
    """40 频带、25 ms 窗、10 ms 帧移的对数梅尔谱，定长 [LOGMEL_FRAMES, 40]。"""
    import librosa
    if len(seg16) < FRAME:
        seg16 = np.pad(seg16, (0, FRAME - len(seg16)))
    mel = librosa.feature.melspectrogram(y=seg16, sr=SR, n_fft=FRAME, hop_length=HOP, n_mels=N_MELS)
    lm = np.log(mel + 1e-6).T
    out = np.zeros((LOGMEL_FRAMES, N_MELS), np.float32)
    k = min(LOGMEL_FRAMES, len(lm))
    out[:k] = lm[:k]
    return out


# ---------------------------------------------------------------------------- wav2vec 2.0
class Wav2VecExtractor:
    """wav2vec 2.0-base：话语内帧级特征（20 ms 步长）的均值池化。"""

    def __init__(self, name: str = "facebook/wav2vec2-base", device: str = "cuda", chunk_sec: float = 20.0):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = Wav2Vec2Model.from_pretrained(name).to(self.device).eval()
        self.chunk = int(chunk_sec * SR)

    def __call__(self, seg16: np.ndarray) -> np.ndarray:
        torch = self.torch
        if len(seg16) < 400:                                  # 不足一个感受野
            return np.zeros(768, np.float32)
        feats, wts = [], []
        with torch.no_grad():
            for s in range(0, len(seg16), self.chunk):
                piece = seg16[s:s + self.chunk]
                if len(piece) < 400:
                    continue
                x = self.fe(piece, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
                h = self.model(x).last_hidden_state[0]
                feats.append(h.mean(0).float().cpu().numpy())
                wts.append(h.size(0))
        if not feats:
            return np.zeros(768, np.float32)
        return np.average(np.stack(feats), axis=0, weights=wts).astype(np.float32)


# ---------------------------------------------------------------------------- 整条录音
def process_recording_audio(path: str, segments: Sequence[Tuple[float, float]], w2v: Wav2VecExtractor,
                            want_logmel: bool = True, noise_snr_db: Optional[float] = None,
                            rng: Optional[np.random.Generator] = None) -> Dict:
    """对每个话语区间计算 wav2vec 特征、保真度指标、行为指标、对数梅尔谱。

    noise_snr_db 不为 None 时先对整段波形加白噪声（相对语音帧功率的 SNR），用于鲁棒性扰动实验。
    VAD 在（可能加噪的）整段波形上运行一次，再按话语区间取帧标记。
    """
    wav, native, nsr = load_audio(path)
    if noise_snr_db is not None:
        rng = rng or np.random.default_rng(0)
        sp = dual_threshold_vad(wav)
        fr = frame_signal(wav, FRAME, HOP)
        k = min(len(fr), len(sp))
        p_sig = float((fr[:k][sp[:k]] ** 2).mean()) if sp[:k].any() else float((wav ** 2).mean())
        noise_std = np.sqrt(p_sig / (10 ** (noise_snr_db / 10)))
        noise = rng.normal(0, noise_std, size=wav.shape).astype(np.float32)
        wav = wav + noise
        if nsr != SR:
            noise_native = np.interp(np.linspace(0, len(noise) - 1, len(native)), np.arange(len(noise)), noise)
        else:
            noise_native = noise
        native = np.clip(native + noise_native.astype(np.float32), -1.0, 1.0)
    speech = dual_threshold_vad(wav)
    n = len(segments)
    out = {"wav2vec": np.zeros((n, 768), np.float32), "fid": np.full((n, 3), np.nan),
           "voiced": np.full(n, np.nan), "energy": np.full(n, np.nan),
           "present": np.zeros(n, bool), "logmel": np.zeros((n, LOGMEL_FRAMES, N_MELS), np.float16) if want_logmel else None}
    for i, (s, e) in enumerate(segments):
        a, b = int(s * SR), int(e * SR)
        seg = wav[a:b]
        if len(seg) < FRAME:                                  # 技术性缺失：该话语无音频
            continue
        out["present"][i] = True
        fs, fe = int(s * 100), int(e * 100)
        frames = speech[fs:fe]
        nat = native[int(s * nsr):int(e * nsr)]
        out["wav2vec"][i] = w2v(seg)
        out["fid"][i] = utterance_fidelity(seg, nat, frames)
        out["voiced"][i], out["energy"][i] = utterance_behavior_audio(seg, frames)
        if want_logmel:
            out["logmel"][i] = log_mel(seg).astype(np.float16)
    return out
