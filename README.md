# Audio Keyword Recognition

基于 Google Speech Commands 数据集的语音关键词识别训练项目。

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
├── collect_data.py         # 数据采集（单进程，支持 --vad）
├── multi_collect_data.py   # 数据采集（多进程并行）
│
├── train_resnet.py          # [可选] ResNet 独立入口
├── train_cnn.py             # [可选] CNN 独立入口
├── eval_resnet.py           # 测试集评估（支持 cnn/resnet）
├── confusion_matrix.py      # 混淆矩阵可视化
│
├── analyzer.exe             # MFCC 特征提取器（C++，来自 D:\audio_analyzer）
├── speech_commands/         # 数据集（30 个类别）
├── checkpoints/             # 训练检查点
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

### 3. 采集特征（如已有 X_features.npy 可跳过）

```bash
# 单进程
python collect_data.py

# 多进程加速
python multi_collect_data.py

# 启用 VAD 静音裁剪（需安装 webrtcvad）
python collect_data.py --vad
```

### 4. 训练模型

```bash
# 统一入口（推荐）
python train.py --model resnet --epochs 60
python train.py --model cnn --epochs 80 --batch-size 64
python train.py --model resnet --class-weights --label-smoothing 0.1

# 断点续训
python train.py --model resnet --resume checkpoints/epoch_0020.pth

# Delta-Delta 特征
python train.py --model resnet --delta-delta
```

### 5. 评估与可视化

```bash
python eval_resnet.py --model resnet --weights audio_resnet_30class.pth
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

| 模型 | 命令 | 说明 |
|------|------|------|
| ResNet | `python train.py --model resnet` | 1D-ResNet-18 风格，4 层残差块，30 类 |
| CNN | `python train.py --model cnn` | 3 层 1D 卷积 + 全局平均池化 |
| MLP | `python train.py --model mlp` | 全连接网络（快速实验） |

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

## 特征管道

```
WAV → analyzer.exe → 13 MFCC + 13 Delta → pad/trunc to 40 frames
  → 先 Split 再在 train 上 fit StandardScaler → PyTorch Tensor (B, 26, 40)
  → 可选 delta-delta → (B, 39, 40)
  → 训练时在线 SpecAugment
```
