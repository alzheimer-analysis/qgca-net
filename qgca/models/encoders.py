"""三个模态的单模态时序编码器。全部输出 H_M ∈ R^{B×n×d}，d = 256。

文本 : BERT-base / 中文 RoBERTa-wwm-ext / XLM-R 的 [CLS]（仅微调最后两层）-> 4 层 8 头 Transformer
       （时间间隔感知相对位置编码）；备选 2 层 Bi-LSTM
语音 : 话语内均值池化的 wav2vec 2.0 帧特征 -> GRU；备选 对数梅尔谱 2 层 CNN + GRU
视频 : [OpenFace(17 AU + 3 姿态 + 2 视线) ; ViT 嵌入 或 HOG/3D 地标的线性投影] -> Transformer；备选 AU/姿态 + GRU
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .transformer import TimeGapTransformer


def run_rnn(rnn: nn.Module, x: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    """对 [B,n,*] 的补零序列做 (Bi)RNN，返回 [B,n,out]。"""
    lengths = seq_mask.sum(1).long().clamp(min=1).cpu()
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    out, _ = rnn(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
    return out * seq_mask.unsqueeze(-1)


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        t, d = cfg.model.text, cfg.model.d
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(t.backbone)
        self.hidden = self.backbone.config.hidden_size
        self._freeze(int(t.finetune_last_layers))
        self.proj = nn.Sequential(nn.Linear(self.hidden, d), nn.LayerNorm(d), nn.Dropout(cfg.model.ffn_dropout))
        self.kind = t.encoder
        if self.kind == "transformer":
            self.temporal = TimeGapTransformer(d, t.n_heads, t.n_layers, cfg.model.ffn_mult, cfg.model.ffn_dropout)
        else:                                                    # 备选：两层 Bi-LSTM
            self.temporal = nn.LSTM(d, d // 2, num_layers=t.lstm_layers, batch_first=True, bidirectional=True,
                                    dropout=cfg.model.ffn_dropout if t.lstm_layers > 1 else 0.0)

    def _freeze(self, k: int):
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = self.backbone.encoder.layer
        for layer in layers[len(layers) - k:]:
            for p in layer.parameters():
                p.requires_grad = True

    def pretrained_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, batch):
        B, n = batch["seq_mask"].shape
        d = self.proj[0].out_features
        text = batch["text"]
        cls = batch["seq_mask"].new_zeros(B, n, self.hidden)
        if text["ids"].size(0) > 0:
            out = self.backbone(input_ids=text["ids"], attention_mask=text["attn"]).last_hidden_state[:, 0]
            cls = cls.clone()
            cls[text["bidx"], text["uidx"]] = out.to(cls.dtype)
        x = self.proj(cls)
        if self.kind == "transformer":
            return self.temporal(x, batch["time"], batch["seq_mask"])
        return run_rnn(self.temporal, x, batch["seq_mask"])


class SpeechEncoder(nn.Module):
    def __init__(self, cfg, kind: str = None):
        super().__init__()
        s, d = cfg.model.speech, cfg.model.d
        self.kind = kind or s.encoder
        if self.kind == "wav2vec_gru":
            self.inp = nn.Sequential(nn.Linear(s.in_dim, d), nn.LayerNorm(d), nn.Dropout(cfg.model.ffn_dropout))
        else:                                                    # 对数梅尔谱：2 层 CNN 得到话语向量
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, s.n_mels // 4)))
            self.inp = nn.Sequential(nn.Linear(32 * (s.n_mels // 4), d), nn.LayerNorm(d), nn.Dropout(cfg.model.ffn_dropout))
        self.gru = nn.GRU(d, d // 2, num_layers=s.gru_layers, batch_first=True, bidirectional=True,
                          dropout=cfg.model.ffn_dropout if s.gru_layers > 1 else 0.0)

    def utterance_vectors(self, batch):
        if self.kind == "wav2vec_gru":
            return self.inp(batch["wav2vec"])
        mel = batch["logmel"]                                    # [B,n,Tm,F]
        B, n, Tm, Fm = mel.shape
        h = self.cnn(mel.reshape(B * n, 1, Tm, Fm))
        return self.inp(h.reshape(B * n, -1)).view(B, n, -1)

    def forward(self, batch):
        return run_rnn(self.gru, self.utterance_vectors(batch), batch["seq_mask"])


class VideoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v, d = cfg.model.video, cfg.model.d
        self.of_dim, self.use_app = int(v.of_dim), bool(v.use_app)
        self.kind = v.encoder
        in_dim = self.of_dim
        if self.use_app and self.kind == "vit_transformer":
            # ViT 嵌入（768）直接使用；DAIC-WOZ 的 HOG+3D 地标先线性投影到同一维度
            self.app_proj = nn.Identity() if int(v.app_dim) == int(v.app_proj_dim) else nn.Linear(v.app_dim, v.app_proj_dim)
            in_dim += int(v.app_proj_dim)
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.Dropout(cfg.model.ffn_dropout))
        if self.kind == "vit_transformer":
            self.temporal = TimeGapTransformer(d, v.n_heads, v.n_layers, cfg.model.ffn_mult, cfg.model.ffn_dropout)
        else:                                                    # 备选：AU/姿态 + GRU
            self.temporal = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)

    def forward(self, batch):
        x = batch["video"]
        if self.kind != "vit_transformer" or not self.use_app:
            x = x[..., :self.of_dim]
        else:
            x = torch.cat([x[..., :self.of_dim], self.app_proj(x[..., self.of_dim:])], -1)
        x = self.inp(x)
        if self.kind == "vit_transformer":
            return self.temporal(x, batch["time"], batch["seq_mask"])
        return run_rnn(self.temporal, x, batch["seq_mask"])


class EncoderBank(nn.Module):
    """按 cfg.model.modalities 构建各模态编码器，返回 {name: H}。"""

    def __init__(self, cfg):
        super().__init__()
        self.names = list(cfg.model.modalities.keys())
        self.groups = {k: v["group"] for k, v in cfg.model.modalities.items()}
        enc = {}
        for name, spec in cfg.model.modalities.items():
            kind = spec["kind"]
            if kind == "text":
                enc[name] = TextEncoder(cfg)
            elif kind == "wav2vec":
                enc[name] = SpeechEncoder(cfg, "wav2vec_gru")
            elif kind == "logmel":
                enc[name] = SpeechEncoder(cfg, "logmel_cnn_gru")
            elif kind == "video":
                enc[name] = VideoEncoder(cfg)
            else:
                raise ValueError(kind)
        self.enc = nn.ModuleDict(enc)

    def pretrained_parameters(self):
        out = []
        for e in self.enc.values():
            if isinstance(e, TextEncoder):
                out += e.pretrained_parameters()
        return out

    def forward(self, batch):
        return {n: self.enc[n](batch) for n in self.names}
