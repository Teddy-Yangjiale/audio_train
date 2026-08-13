# Audio Keyword Recognition

基于 Google Speech Commands 数据集的语音关键词识别训练项目。

30 类、官方说话人不相交划分下的测试集结果（完整对比见 [BENCHMARK.md](BENCHMARK.md)）：

| 模型 | Top-1 | 参数量 | 单条延迟 |
|------|-------|--------|----------|
| 1D-ResNet | 96.12% | 1.90M | 2.39 ms |
| DS-CNN | 95.49% | 80.7K | 1.16 ms |

> **两个必读的评测约定**
> - 精度必须在 `--speaker-split`（Warden 2018 官方说话人哈希划分）下报告。按样本随机划分会让同一说话人同时出现在训练和测试集，虚高约 22 个百分点。
> - 精度必须用 `--features torch`。默认的 `analyzer.exe` 前端每秒只有 8 帧（125 ms 帧移），会压低约 21 个百分点。

## 目录结构

```
audio_train/
├── config.py              # 全局配置
├── data_utils.py           # 数据管线（加载、切分、标准化）
├── models.py               # 模型定义（ResNet / CNN / MLP）
├── trainer.py              # 训练器（Early Stopping / Checkpoint / Resume / 梯度裁剪）
├── augmentation.py         # 数据增强（SpecAugment / 时间偏移等）
├── train.py                # 统一训练入口
├── inference.py            # 单条音频推理
├── export_onnx.py          # 导出 ONNX 模型
├── visualization.py        # t-SNE 可视化
├── data_stats.py           # 数据集统计
│
├── collect_data.py         # 数据采集（analyzer.exe 前端，单进程，支持 --vad）
├── multi_collect_data.py   # 数据采集（analyzer.exe 前端，多进程并行）
├── collect_data_torch.py   # 数据采集（torchaudio 前端，25ms 窗 / 10ms 帧移，推荐）
├── benchmark.py            # 精度 + 参数量 + 延迟 + 吞吐量统一评测
│
├── train_resnet.py          # [可选] ResNet 独立入口
├── train_cnn.py             # [可选] CNN 独立入口
├── eval_resnet.py           # 测试集评估（支持 cnn/resnet）
├── confusion_matrix.py      # 混淆矩阵可视化
│
├── analyzer.exe             # MFCC 特征提取器（C++，来自 D:\audio_analyzer）
├── speech_commands/         # 数据集（30 个类别）
├── checkpoints/             # 训练检查点
├── bandwidth_extension/     # 频带拓展子模块（AudioUNet，8k → 16k）
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集统计（可选）

```bash
python data_stats.py
```

### 3. 采集特征

```bash
# torchaudio 前端（推荐）→ X_features_torch.npy，101 帧
python collect_data_torch.py --workers 4

# analyzer.exe 前端 → X_features.npy，40 帧（其中仅约 8 帧非零）
python multi_collect_data.py
python collect_data.py --vad          # 启用 VAD 静音裁剪（需安装 webrtcvad）
```

两套特征可以共存，训练/评估时用 `--features {torch,analyzer}` 选择，默认 `analyzer`。

### 4. 训练模型

```bash
# 推荐配置：torchaudio 特征 + 说话人不相交划分
python train.py --model dscnn --features torch --speaker-split --epochs 40 --label-smoothing 0.1

python train.py --model resnet --features torch --speaker-split --epochs 40
python train.py --model cnn --epochs 80 --batch-size 64
python train.py --model resnet --class-weights --label-smoothing 0.1

# 断点续训
python train.py --model resnet --resume checkpoints/epoch_0020.pth

# 并发训练多个模型时用独立 checkpoint 目录，避免互相覆盖 best_model.pth
python train.py --model dscnn --checkpoint-dir checkpoints_dscnn

# Delta-Delta 特征
python train.py --model resnet --delta-delta
```

### 5. 评估与可视化

```bash
# 精度 / 参数量 / 延迟 / 吞吐量一次测全，输出 Markdown 表格
python benchmark.py --features torch --speaker-split \
    --weights resnet=audio_resnet_30class_torch.pth dscnn=audio_dscnn_30class_torch.pth

python eval_resnet.py --model dscnn --features torch --speaker-split \
    --weights audio_dscnn_30class_torch.pth
python confusion_matrix.py --model resnet --weights audio_resnet_30class.pth
python visualization.py --model resnet --weights audio_resnet_30class.pth
```

### 6. 单条推理

```bash
python inference.py path/to/audio.wav --model resnet --weights audio_resnet_30class.pth
```

### 7. 导出 ONNX

```bash
python export_onnx.py --model resnet --weights audio_resnet_30class.pth --output model.onnx
```

## 可用模型

| 模型 | 命令 | 参数量 | 说明 |
|------|------|--------|------|
| ResNet | `python train.py --model resnet` | 1.90M | 1D-ResNet-18 风格，4 层残差块 |
| DS-CNN | `python train.py --model dscnn` | 80.7K | 深度可分离卷积 + 残差，精度接近 ResNet |
| CNN | `python train.py --model cnn` | 166K | 3 层 1D 卷积 + 全局平均池化 |
| MLP | `python train.py --model mlp` | 133K | 全连接网络（无卷积基线，输入展平） |

## 高级训练选项

```bash
python train.py --model resnet \
    --epochs 80 \
    --batch-size 256 \
    --class-weights \          # 自动计算逆频率类别权重
    --label-smoothing 0.1 \    # 标签平滑，减轻过拟合
    --scheduler cosine \       # Cosine 退火（默认 plateau）
    --delta-delta \            # 39 维特征（含加速度）
    --no-augment \             # 禁用数据增强
    --resume checkpoints/epoch_0040.pth  # 断点续训
```

## 配置

所有超参数在 `config.py` 中集中管理：

- `BATCH_SIZE`, `EPOCHS`, `LR` — 训练超参
- `LABEL_SMOOTHING` — 标签平滑系数（默认 0.0）
- `USE_CLASS_WEIGHTS` — 是否启用类别权重
- `SCHEDULER` — 学习率调度器 ("plateau" / "cosine")
- `EARLY_STOP_PATIENCE` — 早停耐心值
- `GRAD_CLIP` — 梯度裁剪阈值
- `VAL_SPLIT` / `TEST_SPLIT` — 验证/测试集比例
- `RANDOM_SEED` — 随机种子（保证可复现）
- `FEATURE_SETS` / `use_feature_set()` — 特征集切换（`analyzer` / `torch`）

## 特征管道

```
WAV → 前端提取 → 13 MFCC + 13 Delta → pad/trunc 到固定帧数
  → 先 Split 再在 train 上 fit StandardScaler → PyTorch Tensor (B, 26, T)
  → 可选 delta-delta → (B, 39, T)
  → 训练时在线 SpecAugment
```

两种前端的差异：

| 前端 | 窗长 / 帧移 | 帧数 | 每条实际有效帧 |
|------|-------------|------|----------------|
| `analyzer.exe` | — / 125 ms | 40 | **7.88**（其余为零填充） |
| `torchaudio` | 25 ms / 10 ms | 101 | **99.4** |

analyzer.exe 的时间分辨率约为 KWS 文献标准做法的 1/12，是精度的主要瓶颈，
新代码请使用 `--features torch`。

## 数据划分

`data_utils.py` 提供两种划分：

- 默认：按样本随机划分。**同一说话人会同时出现在训练和测试集**，测得的精度虚高约 22 个百分点，不可用于对外报告。
- `--speaker-split`：Warden 2018 §7 的官方说话人哈希划分（`which_set()`），与已发表结果同协议。

无论哪种划分，StandardScaler 都只在训练集上 fit，避免统计量泄漏。

## 频带拓展子模块

`bandwidth_extension/` 用 1D U-Net(AudioUNet) 把 8 kHz 音频重建到 16 kHz。
自监督构造训练对：原始 16 kHz 音频为 target，降采样再升采样回来的版本为 input，
模型只需补出丢失的高频。损失为 `100 × L1(时域) + 多分辨率 STFT 损失`（FFT 2048/1024/512）。

```bash
cd bandwidth_extension
python train_bwe.py --limit 3000 --epochs 24 --batch-size 16 --lr 0.0001
python benchmark_bwe.py --model audio_unet_bwe.pth --limit 600   # 对比朴素上采样基线
python inference_bwe.py input.wav output.wav --model audio_unet_bwe.pth
```

评测务必与朴素上采样基线对比（`benchmark_bwe.py` 会自动输出该行）：
本数据集高频能量很低，"什么都不做"本身就有 29.45 dB SNR，
单看模型的 SNR 绝对值会严重误判效果。当前模型 LSD 8.83 dB（基线 23.85 dB），
SNR 27.96 dB（基线 29.45 dB）—— 谱保真度大幅改善，波形保真度略低于基线。
