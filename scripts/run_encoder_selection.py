"""单模态编码器选择（Table: Encoder Selection by Cross-Validation Within Training Data）。

只使用训练数据：DAIC-WOZ 训练集 5 折 CV；MPSC 第一个外层折的训练集内层 CV。比较:
  文本: BERT/RoBERTa + Bi-LSTM  vs  + 时间间隔感知 Transformer
  语音: 对数梅尔 CNN+GRU        vs  wav2vec 2.0 + GRU
  视频: AU/姿态 + GRU           vs  ViT(或 HOG/地标) + Transformer
每个单模态模型 = 编码器 + 注意力池化 + 头（无门控、无跨模态注意力）。输出宏 F1 与 AUC（受试者级折外预测）。
"""
import argparse
import os

from _common import add_common_args, build_cfg, ensure_dir
from qgca.config import update_cfg
from qgca.data.splits import mpsc_nested_splits
from qgca.engine.trainer import aggregate_subjects
from qgca.evaluation import metrics as M
from qgca.models.factory import prepare_model_cfg
from qgca.protocols import common as C
from qgca.protocols.daic import daic_folds
from qgca.utils import get_logger, save_json
import numpy as np

VARIANTS = {
    "text_bilstm": {"model.modalities": {"T": {"kind": "text", "group": "T"}}, "model.text.encoder": "bilstm"},
    "text_transformer": {"model.modalities": {"T": {"kind": "text", "group": "T"}}, "model.text.encoder": "transformer"},
    "speech_logmel_cnn_gru": {"model.modalities": {"A": {"kind": "logmel", "group": "A"}}},
    "speech_wav2vec_gru": {"model.modalities": {"A": {"kind": "wav2vec", "group": "A"}}},
    "video_au_gru": {"model.modalities": {"V": {"kind": "video", "group": "V"}}, "model.video.encoder": "au_gru"},
    "video_vit_transformer": {"model.modalities": {"V": {"kind": "video", "group": "V"}}, "model.video.encoder": "vit_transformer"},
}
UNIMODAL_COMMON = {"model.gate_mode": "none", "model.fusion": "concat", "model.modality_dropout": 0.0, "loss.lambda_ent": 0.0}


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    base = build_cfg(args)
    out = args.out or os.path.join(base.output_dir, base.dataset, "encoder_selection")
    ensure_dir(out)
    get_logger("qgca", out + "/log.txt")
    index = C.load_index(base.data.processed_dir)
    epochs = args.epochs or sorted(base.train.epochs_grid)[len(base.train.epochs_grid) // 2]
    if base.dataset == "daic_woz":
        entries = C.select_entries(index, ["train"])
        folds = daic_folds(entries, base.protocol.cv_folds)
    else:
        tab = C.subject_table(index)
        subs = sorted(tab)
        sp0 = mpsc_nested_splits(subs, [tab[s]["label"] for s in subs], [tab[s]["age_group"] for s in subs],
                                 base.protocol.outer_folds, 1, base.protocol.inner_folds)[0]
        entries, folds = C.select_entries(index, subjects=sp0["train"]), sp0["inner"]
    res = {}
    for name, patch in VARIANTS.items():
        cfg = prepare_model_cfg(update_cfg(base, {**UNIMODAL_COMMON, **patch}))
        oof = C.cv_oof(cfg, entries, folds, epochs, C.seeds_of(cfg, selection=True))
        subj = aggregate_subjects(oof)
        y = np.array([s["y"] for s in subj]).astype(int)
        p = np.array([s["prob"] for s in subj])
        res[name] = {"macro_f1": M.macro_f1(y, (p >= 0.5).astype(int)), "auc": M.auc(y, p)}
        print(name, res[name])
    save_json(res, os.path.join(out, "encoder_selection.json"))


if __name__ == "__main__":
    main()
