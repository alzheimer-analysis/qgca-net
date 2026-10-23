"""按 cfg.model.name 构建模型。"""
from copy import deepcopy

from ..config import Cfg
from .baselines import BASELINES
from .qgca_net import QGCANet, build_qgca_speech_cfg


def prepare_model_cfg(cfg) -> Cfg:
    """对基线/语音变体做必要的配置调整（返回新配置）。"""
    c = deepcopy(cfg)
    name = c.model.name
    if name in BASELINES:
        c.model.gate_mode = "none"                    # 基线不使用质量指标，只保留技术性缺失掩码
        c.model.modality_dropout = 0.0
        c.loss.lambda_ent = 0.0
        if "baseline" not in c:
            c["baseline"] = Cfg({"hidden": 256, "lstm_layers": 1, "layers": 2, "heads": 8, "channels": 128,
                                 "lstm_hidden": 256, "shared_dim": 256, "ortho_weight": 0.1})
        if name == "depaudionet":                     # 直接使用对数梅尔谱，不经 EncoderBank
            c.model.modalities = Cfg({})
        if name == "dmpf":                            # 两个声学视角
            c.model.modalities = Cfg({"A1": Cfg({"kind": "wav2vec", "group": "A"}),
                                      "A2": Cfg({"kind": "logmel", "group": "A"})})
    if name == "qgca_speech":
        c = build_qgca_speech_cfg(c)
    return c


def build_model(cfg):
    cfg = prepare_model_cfg(cfg)
    name = cfg.model.name
    if name in ("qgca", "qgca_speech"):
        return QGCANet(cfg), cfg
    if name in BASELINES:
        return BASELINES[name](cfg), cfg
    raise ValueError(f"unknown model: {name}")
