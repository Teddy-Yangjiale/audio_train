# Changelog

## 原始实现（v0）

### 项目概览
基于 Google Speech Commands 数据集的语音关键词识别训练项目。使用 C++ 编写的 `analyzer.exe` 从 WAV 提取 MFCC 特征，
Python 端负责数据采集、模型训练与评估。数据集包含 30 个关键词类别。

### 原始文件清单

| 文件 | 说明 |
|------|------|
| `analyzer.exe` | C++ MFCC 特征提取器（来自 D:\audio_analyzer） |
| `speech_commands/` | 30 类 WAV 数据集 |
| `collect_data.py` | 单进程数据采集，调用 analyzer.exe 提取特征，输出 X_features/y_labels/label_map |
| `multi_collect_data.py` | 多进程并行数据采集，Pool + imap 加速 |
| `train_model.py` | 基线模型：随机森林（4 类） |
| `train_deep_model.py` | 早期 MLP 全连接网络（4 类，39 维输入） |
| `train_cnn.py` | 1D-CNN 模型训练（按 label_map 自适应类别数） |
| `train_resnet.py` | 1D-ResNet 模型训练（30 类） |
| `eval_resnet.py` | ResNet 评估，输出分类报告 |
| `confusion_matrix.py` | ResNet 混淆矩阵可视化 |
| `test.py` | 工具脚本，重新生成 label_map.npy |

### 原始产物

- `X_features.npy` — ~57k 样本 × 40 帧 × 26 维（13 MFCC + 13 Delta）
- `y_labels.npy` — 标签数组
- `label_map.npy` — 类别名映射（30 类）
- `audio_cnn_v1.pth` — 训练好的 CNN 权重
- `audio_resnet_30class.pth` — 训练好的 ResNet 权重
- `confusion_matrix.png` — 混淆矩阵图

### 原始实现已知问题

| # | 问题 | 严重度 |
|---|------|--------|
| 1 | `StandardScaler` 在 split 前 fit 全量数据，测试集信息泄露 | 严重 |
| 2 | 用测试集调优（scheduler、监控），无验证集 | 严重 |
| 3 | `ResidualBlock1D` / `AudioResNet` 在三处 copy-paste 重复定义 | 高 |
| 4 | 训练脚本间预处理逻辑不一致（RF 用原始 3D，MLP 假设 39 维） | 高 |
| 5 | 所有路径、维度、超参数硬编码，无配置文件 | 中 |
| 6 | 无 Early Stopping、无 Model Checkpoint、无梯度裁剪 | 中 |
| 7 | 无数据增强、无完整随机种子固定 | 中 |
| 8 | Delta 计算用 `append=mfccs[-1:]` 边界处理粗糙 | 低 |
| 9 | 无推理脚本、无 ONNX 导出、无 requirements.txt | 低 |

---

## v1.0 — 模块化重构（2026-05-13）

### 新增文件

| 文件 | 说明 |
|------|------|
| `config.py` | 全局配置中心：所有超参数、路径、随机种子集中管理，`Config.set_seed()` 固定全部随机种子 |
| `data_utils.py` | 统一数据管线：`load_raw_features()` 加载数据，`split_and_scale()` **先 split 再 fit scaler**（修复数据泄露），三路划分 train/val/test，`preprocess_single()` 支持推理时单条预处理，`compute_delta_delta()` 按需计算一阶加速度特征 |
| `models.py` | 所有模型定义集中于此：`ResidualBlock1D`、`AudioResNet`、`Audio1DCNN`、`AudioMLP`，提供 `create_model()` 工厂函数和 `load_model_weights()` 加载函数 |
| `trainer.py` | 统一训练器类：`_train_epoch()` 单轮训练含梯度裁剪，`_validate()` 验证集评估，`fit()` 完整训练循环含 Early Stopping + Model Checkpoint + ReduceLROnPlateau，`evaluate()` 输出分类报告，`save_model()` 保存权重，断点续训支持 |
| `augmentation.py` | 数据增强模块：时域掩码 `time_mask()`、频域掩码 `freq_mask()`、时间偏移 `time_shift()`、组合 `spec_augment()` |
| `inference.py` | 单条 WAV 推理：analyzer.exe → 预处理 → 模型预测 → Top-5 结果输出 |
| `export_onnx.py` | PyTorch 模型导出为 ONNX 格式 |
| `requirements.txt` | 依赖声明：torch, numpy, scikit-learn, matplotlib, seaborn, tqdm, joblib |
| `README.md` | 项目文档：目录结构、快速开始、配置说明 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `train_cnn.py` | **重写**：删除了数据加载/标准化/模型定义代码（~130 行），改为调用 data_utils/models/trainer（~30 行）。验证集监控替代测试集泄露 |
| `train_resnet.py` | **重写**：同上，精简为入口脚本。使用 AdamW + ReduceLROnPlateau + 梯度裁剪 |
| `eval_resnet.py` | **重写**：模型定义从 models.py 导入，不再重复定义；数据加载改为 data_utils 管线；评估报告写入 `evaluation_report.txt` |
| `confusion_matrix.py` | **重写**：模型定义从 models.py 导入；数据加载改为 data_utils 管线 |
| `collect_data.py` | **修改**：Delta 计算从 `append=mfccs[-1:]` 改为 `prepend=mfccs[:1]`；路径常量改为从 Config 导入 |
| `multi_collect_data.py` | **修改**：同上，delta 边界处理改进，路径统一到 Config |

### 未修改文件（保留原样）

| 文件 | 原因 |
|------|------|
| `analyzer.exe` | 外部产物，不动 |
| `speech_commands/` | 原始数据集，不动 |
| `train_model.py` | 早期 RF 基线实验，保留做参考 |
| `train_deep_model.py` | 早期 MLP 实验，保留做参考 |
| `test.py` | 原有功能被 data_utils 覆盖，保留但不建议使用 |

### 核心修复

1. **数据泄露修复**：`StandardScaler` 现在严格仅在训练集上 fit，然后 transform val/test
2. **三路划分**：train 80% / val 10% / test 10%，val 集用于早停和调度器，test 集仅最终评估时使用一次
3. **可复现性**：`Config.set_seed()` 固定 random / numpy / torch / cuda 所有种子
4. **代码去重**：`ResidualBlock1D` 和 `AudioResNet` 从三处 copy-paste 收归 models.py 单一定义
5. **Delta 边界优化**：`np.diff(mfccs, axis=0, prepend=mfccs[:1])` 替代 `append=mfccs[-1:]`

### 验证结果

- 数据管线：train=45897, val=5738, test=5738, 30 类正确加载
- 模型构建：CNN 165,982 参数正常
- 训练循环：5 epoch CNN 达 72.95% test accuracy，early stopping / checkpoint / 梯度裁剪全部正常工作

---

## v1.1 — 增强管线 + 分析工具（2026-05-13）

### 新增文件

| 文件 | 说明 |
|------|------|
| `visualization.py` | t-SNE 特征可视化：提取模型倒数第二层 embeddings → 2D 降维 → 散点图 |
| `CHANGELOG.md` | 本文件，记录所有版本变更 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `trainer.py` | 修复 `numpy` import 位置（从文件底部移到顶部）；新增 `AugmentedTensorDataset` 类，包装训练数据按 50% 概率在线应用 SpecAugment；`build_loaders()` 新增 `augment` 参数 |
| `train_cnn.py` | 添加 `argparse`：`--epochs`、`--batch-size`、`--no-augment`、`--delta-delta`；默认启用数据增强；自动适配 delta-delta 时的 in_channels |
| `train_resnet.py` | 同上，添加 `argparse` 支持，默认启用增强 |
| `eval_resnet.py` | 添加 `argparse`：`--model`、`--weights`、`--delta-delta`；新增 `top_k_accuracy()` 计算 Top-3 准确率；新增 `most_confused_pairs()` 分析最易混淆词对；报告同时输出到控制台和 `evaluation_report.txt` |
| `confusion_matrix.py` | 添加 `argparse`：`--model`、`--weights`、`--delta-delta`、`--output`；数值显示改为百分比 `.0f`；添加单元格分隔线 |
| `export_onnx.py` | 添加 `argparse` 支持；新增 Windows GBK 编码修复（`PYTHONIOENCODING=utf-8`）；opset 升级到 18 |
| `models.py` | `load_model_weights()` 兼容两种格式：完整 checkpoint dict（含 `model_state_dict` 键）和纯 `state_dict` |
| `data_utils.py` | `split_and_scale()` 新增 `use_delta_delta` 路径：在 split 前计算 39 维特征（13 MFCC + 13 Delta + 13 Delta-Delta）；`preprocess_single()` 新增 `use_delta_delta` 参数 |
| `requirements.txt` | 补充 `onnx` 和 `onnxscript` 依赖 |

### 功能增强

1. **在线数据增强**：训练时 50% 概率应用 SpecAugment（时域+频域掩码），val/test 不做增强
2. **Delta-Delta 特征**：通过 `--delta-delta` 参数启用，特征维度从 26 → 39，按需计算无需重新采集
3. **Top-3 准确率**：评估时同时报告 Top-1 和 Top-3 准确率
4. **易混淆词对分析**：自动找出测试集中最易互相误判的词对（如 go↔no, three↔tree）
5. **t-SNE 可视化**：可视化模型倒数第二层特征嵌入在 2D 空间中的分布
6. **命令行接口统一**：所有训练/评估/推理脚本均支持 `argparse`

### 验证结果

- 增强管线：`AugmentedTensorDataset` 正确应用 SpecAugment，val loader 保持原始数据
- 评估增强：Top-1 73%，Top-3 90%，混淆对 "go ↔ no" (84 errors) 符合预期
- ONNX 导出：成功导出，opset 18
- 推理脚本：WAV → 预测 Top-5 概率正常

### 当前目录结构

```
audio_train/
├── config.py              # 全局配置
├── data_utils.py           # 数据管线
├── models.py               # 模型定义
├── trainer.py              # 训练器
├── augmentation.py         # 数据增强
├── inference.py            # 推理
├── export_onnx.py          # ONNX 导出
├── visualization.py        # t-SNE 可视化
│
├── collect_data.py         # 数据采集（单进程）
├── multi_collect_data.py   # 数据采集（多进程）
├── train_resnet.py          # ResNet 训练入口
├── train_cnn.py             # CNN 训练入口
├── eval_resnet.py           # 评估（支持 cnn/resnet）
├── confusion_matrix.py      # 混淆矩阵
│
├── analyzer.exe             # C++ MFCC 提取器
├── speech_commands/         # 数据集
├── checkpoints/             # 训练检查点
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

---

## v1.2 — 统一训练 + 高级训练特性（2026-05-13）

### 新增文件

| 文件 | 说明 |
|------|------|
| `train.py` | **统一训练入口**（替代 `train_cnn.py` / `train_resnet.py` 分散脚本）。通过 `--model resnet/cnn/mlp` 切换模型，支持所有 CLI 参数 |
| `data_stats.py` | 数据集统计与分析：每类样本数、均值/标准差、值域，生成 `dataset_stats.png` |

### 修改文件

| 文件 | 变更 |
|------|------|
| `config.py` | 新增 `LABEL_SMOOTHING`(0.0)、`USE_CLASS_WEIGHTS`(False)、`SCHEDULER`("plateau")、`COSINE_T_MAX`(100)、`SAVE_BEST_ONLY`(True) |
| `trainer.py` | **重大更新**：`__init__` 支持 `class_weights`（类别权重）、`label_smoothing`（标签平滑）、`scheduler_type`("plateau"/"cosine")；`fit()` 新增 `resume` 参数支持**断点续训**；日志增强（显示 LR、scheduler 类型、权重状态）；新增 `_resume()` 方法恢复 optimizer/scheduler/epoch 状态；训练 DataLoader 新增 `drop_last=True` 避免最后 batch 归一化问题；`_compute_class_weights()` 工具函数根据训练集分布自动计算逆频率权重 |
| `collect_data.py` | 新增可选 `--vad` 参数（需 `webrtcvad`），自动去除首尾静音段；重构为模块化函数 `extract_mfcc/pad_or_truncate/build_feat_matrix`；`SAMPLES_PER_CLASS` 集中为变量 |
| `multi_collect_data.py` | `SAMPLES_PER_CLASS` 集中为变量，代码风格统一 |

### `train.py` 命令行参数完整列表

```
python train.py --model resnet|cnn|mlp
                [--epochs N] [--batch-size N] [--lr FLOAT]
                [--no-augment] [--delta-delta] [--class-weights]
                [--label-smoothing FLOAT] [--scheduler plateau|cosine]
                [--resume checkpoints/xxx.pth] [--output model.pth]
```

### 功能增强

1. **统一训练入口**：一个 `train.py` 替代分散的 `train_cnn.py` / `train_resnet.py`，参数全覆盖
2. **类别权重**：`--class-weights` 自动按逆频率计算权重，缓解类别不均衡（如 bed=1713 vs down=2000）
3. **标签平滑**：`--label-smoothing 0.1` 将 hard label 软化为 0.9/0.1，减轻过拟合
4. **Cosine 退火**：`--scheduler cosine` 替代 Plateau 调度器，适用于固定 epoch 训练
5. **断点续训**：`--resume checkpoints/epoch_0020.pth` 恢复 optimizer/scheduler 状态继续训练
6. **数据集统计**：`data_stats.py` 一键分析每类样本分布、特征值域、均值/标准差

### 验证结果

- `train.py --model cnn --epochs 3 --batch-size 128` → 68% test acc ✓
- `train.py --class-weights --label-smoothing 0.1 --scheduler cosine` → 正常运行 ✓
- `train.py --resume checkpoints/best_model.pth` → 从 epoch 2 正确恢复到 epoch 3 ✓
- `data_stats.py` → 输出完整分布报告 + 柱状图 ✓

### 当前目录结构

```
audio_train/
├── config.py              # 全局配置
├── data_utils.py           # 数据管线
├── models.py               # 模型定义
├── trainer.py              # 训练器（class_weights/label_smoothing/resume）
├── augmentation.py         # 数据增强
├── train.py                # 统一训练入口（--model resnet/cnn/mlp）
├── inference.py            # 推理
├── export_onnx.py          # ONNX 导出
├── visualization.py        # t-SNE 可视化
├── data_stats.py           # 数据集统计
│
├── collect_data.py         # 数据采集（单进程，支持 --vad）
├── multi_collect_data.py   # 数据采集（多进程）
├── train_cnn.py             # [保留] 原 CNN 入口，可被 train.py 替代
├── train_resnet.py          # [保留] 原 ResNet 入口，可被 train.py 替代
├── eval_resnet.py           # 评估（支持 cnn/resnet）
├── confusion_matrix.py      # 混淆矩阵
│
├── analyzer.exe             # C++ MFCC 提取器
├── speech_commands/         # 数据集
├── checkpoints/             # 训练检查点
├── requirements.txt
├── README.md
└── CHANGELOG.md
```
