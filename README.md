# QGCA-Net 复现代码

论文：基于音视频与文本多模态时序协同建模的青少年抑郁症辅助诊断研究（Quality-aware Gated Cross-modal Attention Network, QGCA-Net；
英文稿见 `../IEEE_Access_paper/main.tex`）。本工程按论文方法与实验协议实现，**不包含绘图脚本与数值核对脚本**，且尚未运行过（需自备数据与预训练权重）。

## 方法 → 代码对照

| 论文章节 | 内容 | 代码 |
|---|---|---|
| Problem Formulation | 话语级 (T,A,V,𝓜) → (ŷ, r̂) | `qgca/data/dataset.py`, `qgca/models/qgca_net.py` |
| Unimodal Temporal Encoders | BERT/RoBERTa/XLM-R(末两层微调)+4 层 8 头 Transformer(时间间隔相对位置编码)；wav2vec2+GRU；OpenFace+ViT(DAIC: HOG/3D 地标投影)+Transformer；备选 Bi-LSTM / 对数梅尔 CNN-GRU / AU-GRU | `qgca/models/encoders.py`, `qgca/models/transformer.py` |
| Signal-Fidelity Indicators | 7 个保真度指标、训练集 min-max、凸组合 q、掩码 `m = 技术性存在 ∧ q≥τ` | `qgca/preprocessing/{audio,video,text}.py`, `qgca/data/quality.py`, `qgca/data/normalization.py` |
| Quality-Aware Gating | `g=σ(W2 ReLU(W1[h;q]))`，`h'=g·m·h`，门控熵正则 | `qgca/models/gating.py` |
| Bidirectional Cross-Modal Attention | 6 方向、8 头、2 层、被掩码键权重为 0、`LN(H'+ΣDropout(CA))`、FFN 融合 | `qgca/models/cross_attention.py` |
| Attentive Pooling & Multi-Task Heads | 注意力池化；风险头；数据集专属严重度头(目标缩放到[0,1]) | `qgca/models/heads.py` |
| Training Objective | Focal + Huber + 熵正则 + 域对抗(GRL)；整序列模态 Dropout 0.15；文本/语音/视频增强；AdamW + 5 epoch 预热余弦 | `qgca/losses.py`, `qgca/engine/trainer.py`, `qgca/data/augment.py` |
| Separation of Model Selection and Evaluation | 所有决策只在训练数据内完成 | `qgca/protocols/{common,daic,avec,mpsc}.py` |
| Metrics and Statistical Analysis | 受试者级、分层 bootstrap CI、Hanley–McNeil、配对 bootstrap + DeLong、Holm、合并分析 | `qgca/evaluation/{metrics,stats}.py`, `scripts/run_stats.py` |
| Compared Methods | 早/晚期融合、DepAudioNet、DMPF、MLAN、MulT、TAMFN、TTFNet、QGCA-Speech | `qgca/models/baselines.py`, `qgca/models/qgca_net.py` |
| Quality-indicator leakage | 指标–标签相关、q-only 分类器、四组对照 | `qgca/evaluation/confound.py`, `scripts/run_confound.py`, `configs/controls.yaml` |
| Ablation / Robustness / Sensitivity / Complexity | | `configs/{ablation,robustness,sensitivity}.yaml`, `scripts/run_{variants,robustness,sensitivity}.py`, `scripts/profile_complexity.py` |
| Cross-corpus / cross-lingual transfer | DSH、DAL、零样本与微调 | `qgca/protocols/transfer.py`, `scripts/run_transfer.py` |

超参数（`configs/default.yaml`）与论文 Table "Hyperparameters of QGCA-Net" 一致：d=256，门控隐层 128，8 头×2 层，
w_A=(0.50,0.25,0.25)，w_V=(0.50,0.25,0.25)，τ=0.1，focal γ=2.0/α_pos=0.70，Huber δ=0.10，
λ=(1.0,0.5,0.05,0.1)，p_md=0.15，dropout 0.1；AdamW(wd 0.01)，学习率 1e-5(预训练)/1e-3(新模块)，batch 16，5 个种子。

## 安装

```bash
pip install -r requirements.txt
```
另需：OpenFace 2.0（`FeatureExtraction`）、HuggingFace 权重（`bert-base-uncased`、`hfl/chinese-roberta-wwm-ext`、`xlm-roberta-base`、
`facebook/wav2vec2-base`、`google/vit-base-patch16-224-in21k`）、Whisper large-v3。

## 流程

```bash
# 1) 预处理 → data/processed/<dataset>/{*.pt, index.json}
python scripts/prepare_daic_woz.py --root <DAIC-WOZ根目录> --labels_dir <标签csv目录> --out data/processed/daic_woz
python scripts/run_openface.py     --manifest avec_manifest.csv --exe <FeatureExtraction>
python scripts/prepare_avec2014.py --manifest avec_manifest.csv --out data/processed/avec2014
python scripts/prepare_mpsc.py     --manifest mpsc_manifest.csv --out data/processed/mpsc

# 2) 训练数据内的模型选择（编码器、超参数；轮数/阈值在主实验里由 CV 自动选择）
python scripts/run_encoder_selection.py --config configs/daic_woz.yaml
python scripts/run_hparam_search.py      --config configs/daic_woz.yaml            # QGCA-Net，24 个配置
python scripts/run_hparam_search.py      --config configs/daic_woz.yaml --extra configs/baselines.yaml --set model.name=mult

# 3) 主实验（留出集只评估一次）
python scripts/run_experiment.py --config configs/daic_woz.yaml --hparams outputs/daic_woz/hparam_qgca/best.json
python scripts/run_experiment.py --config configs/avec2014.yaml
python scripts/run_experiment.py --config configs/mpsc.yaml
python scripts/run_experiment.py --config configs/daic_woz.yaml --extra configs/baselines.yaml --set model.name=ttfnet

# 4) 统计比较、消融、混杂对照、鲁棒性、敏感性、迁移、复杂度
python scripts/run_stats.py --corpus daic --ref qgca --runs qgca=outputs/daic_woz/qgca mult=outputs/daic_woz/mult
python scripts/run_variants.py --config configs/daic_woz.yaml --variants configs/ablation.yaml
python scripts/run_variants.py --config configs/daic_woz.yaml --variants configs/controls.yaml
python scripts/run_confound.py  --config configs/daic_woz.yaml
python scripts/make_perturbed_records.py --processed data/processed/daic_woz --kind speech_noise --snr_db 10 --out data/processed/daic_woz_speech_noise_10db
python scripts/make_perturbed_records.py --processed data/processed/daic_woz --kind video_occlusion --ratio 0.3 --out data/processed/daic_woz_video_occlusion_30
python scripts/run_robustness.py --config configs/daic_woz.yaml --selection outputs/daic_woz/qgca/selection.json \
       --perturbed speech_noise_10db=data/processed/daic_woz_speech_noise_10db video_occlusion_30=data/processed/daic_woz_video_occlusion_30
python scripts/run_sensitivity.py --config configs/daic_woz.yaml
python scripts/run_transfer.py cross_corpus --direction avec_to_daic --dsh 1 --dal 1
python scripts/run_transfer.py cross_lingual --method qgca
python scripts/profile_complexity.py --config configs/daic_woz.yaml
```

## Manifest 格式

* AVEC2014：`record_id,subject_id,task,split(train|dev|test),audio,video,openface,bdi`
* MPSC：`record_id,subject_id,task,audio,video,openface,sync_audio_start,sync_video_start,diagnosis,phq9,age`

## 论文未写明、本实现所作的假设（请在复现实验时知悉）

1. **保真度指标缺失项**：DAIC-WOZ 无原始视频，视频只有跟踪置信度可用；缺失指标在凸组合中按可用指标重新归一权重。DAIC-WOZ 发布的 AU 强度不足 17 个，缺列以 0 补齐。
2. **时间间隔相对位置编码**：用话语起始时间差的符号对数分桶 → 每头一个可学习偏置；视频 Transformer 同样使用（2 层）。
3. **单向注意力消融**：取环形 3 个方向（A→T，V→A，T→V），每个模态恰被一个源查询。
4. **整序列模态 Dropout**：若一个样本三个模态全被丢弃，随机恢复其中一个。
5. **数据增强**：语音增强作用于预提取特征（wav2vec 特征上的高斯噪声；对数梅尔谱上 SpecAugment）；视频增强（时间裁剪与丢帧）作用于话语内的帧序列，池化之前；英文同义词替换默认使用 WordNet，中文需通过 `augment.text.synonym_file` 提供词典。
6. **行为类 q**（仅混杂对照）：六个行为指标 min-max 后分三组（文本：回答长度、情感词率；语音：发声比例、能量；视频：头部运动、视线离散度）取等权凸组合。
7. **基线**：DepAudioNet、DMPF、MLAN、MulT、TAMFN、TTFNet 按其核心思想实现，三模态基线共享 QGCA-Net 的编码器；它们与 QGCA-Net 享有同一 24 配置的调参预算（`configs/search_space.yaml`）。
8. **晚期融合**：投票权重正比于折外的各单模态宏 F1。
9. **迁移**：零样本时的 DAL 仅使用目标域训练录音的输入（无标签）训练域判别器；DAIC→AVEC 零样本回归使用源域严重度头并换算到 BDI-II 量表；微调学习率为原来的 1/10。
10. **判决阈值**：在训练数据折外的受试者级预测上取使宏 F1 最大（并列时最接近 0.5）的阈值；MPSC 取 25 个外层折阈值的均值。
