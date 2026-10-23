"""文本预处理：DAIC-WOZ 人工转写解析、Whisper large-v3 转写（MPSC）、按静默切分话语、ASR 置信度。

文本保真度: 话语内所有词的平均 ASR token 对数概率（人工校对转写不适用，q_T 置 1）。
行为类指标（仅混杂分析）: token 数、情感词率。
"""
import csv
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CJK = re.compile(r"[一-鿿]")

# 极简情感词表；正式实验请通过 affect_lexicon 传入完整词表（如 LIWC / 中文情感词汇本体）
_AFFECT_EN = {"sad", "happy", "angry", "afraid", "anxious", "depressed", "lonely", "tired", "hopeless", "worried",
              "upset", "guilty", "ashamed", "fine", "good", "bad", "great", "terrible", "nervous", "stressed"}
_AFFECT_ZH = {"难过", "开心", "生气", "害怕", "焦虑", "抑郁", "孤独", "累", "绝望", "担心", "沮丧", "内疚", "紧张", "压力",
              "高兴", "痛苦", "烦", "失落", "快乐", "悲伤"}


def load_affect_lexicon(path: Optional[str]) -> set:
    if path is None:
        return _AFFECT_EN | _AFFECT_ZH
    with open(path, "r", encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def tokenize_simple(text: str) -> List[str]:
    return list(text.replace(" ", "")) if _CJK.search(text) else text.lower().split()


def behavior_text(text: str, lexicon: set) -> Tuple[float, float]:
    toks = tokenize_simple(text)
    if not toks:
        return 0.0, 0.0
    if _CJK.search(text):
        hits = sum(text.count(w) for w in lexicon if _CJK.search(w))
    else:
        hits = sum(t in lexicon for t in toks)
    return float(len(toks)), hits / len(toks)


# ---------------------------------------------------------------------------- DAIC-WOZ
def parse_daic_transcript(path: str, speaker: str = "Participant") -> List[Dict]:
    """*_TRANSCRIPT.csv（制表符分隔: start_time, stop_time, speaker, value）。只保留受访者话语。"""
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["speaker"].strip().lower() != speaker.lower():
                continue
            txt = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", row["value"]).strip()   # 去掉 <synch>/[laughter] 之类的标注
            txt = re.sub(r"\s+", " ", txt)
            out.append({"start": float(row["start_time"]), "end": float(row["stop_time"]), "text": txt})
    return out


# ---------------------------------------------------------------------------- Whisper（MPSC）
class WhisperASR:
    """Whisper large-v3，语言固定为中文；返回带词级时间戳与对数概率的词序列。"""

    def __init__(self, model_name: str = "large-v3", language: str = "zh", device: str = None):
        import whisper
        import torch
        self.model = whisper.load_model(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.language = language

    def words(self, wav_path: str) -> List[Dict]:
        res = self.model.transcribe(wav_path, language=self.language, word_timestamps=True, condition_on_previous_text=False)
        out = []
        for seg in res["segments"]:
            for w in seg.get("words", []):
                out.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                            "logprob": float(np.log(max(w.get("probability", 1e-6), 1e-6)))})
        return out


def split_by_silence(words: Sequence[Dict], min_silence: float = 0.5) -> List[Dict]:
    """在词间静默 >= min_silence（秒）处切分话语；保留词级对数概率，话语置信度为其均值。"""
    utts, cur = [], []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] >= min_silence:
            utts.append(cur)
            cur = []
        cur.append(w)
    if cur:
        utts.append(cur)
    return [{"start": u[0]["start"], "end": u[-1]["end"], "text": "".join(x["word"] for x in u).strip(),
             "asr_logprob": float(np.mean([x["logprob"] for x in u]))} for u in utts]
