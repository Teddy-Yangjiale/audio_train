# Benchmark Report

All numbers below were measured on this repository (CPU-only, PyTorch 2.12).
Reproduction commands are given per section.

---

## 1. Evaluation protocol

Two things had to be fixed before any number here could be compared with published work.

**Speaker-disjoint split.** The original pipeline split samples randomly, so the same
speaker appeared in train and test. Speech Commands records many utterances per speaker,
so a model can score well by memorising voices. Published results all use the official
speaker-hash split from Warden 2018 §7, implemented here as `which_set()` in
[data_utils.py](data_utils.py) and enabled with `--speaker-split`.

| 1D-ResNet, analyzer features | Top-1 | Macro-F1 |
|---|---|---|
| Random per-utterance split | 97.23% | 97.24% |
| Speaker-disjoint split | 74.75% | 75.23% |

**22.5 points of the original accuracy came from speaker leakage.** Every number below
uses the speaker-disjoint split.

**LSD definition.** `compute_lsd()` used natural logarithms and took the RMS over the
batch axis instead of the frequency axis, so its values were not comparable to the dB
LSD reported in the literature. Fixed in [losses_bwe.py](bandwidth_extension/losses_bwe.py)
to the standard definition (power spectra, −80 dB floor, RMS over frequency bins).

---

## 2. Keyword spotting — feature pipeline

The `analyzer.exe` front end emits **~8 MFCC frames per 1 s clip** (125 ms hop, 64-point
FFT). Measured over the full dataset: **7.88 non-zero frames out of a 40-frame matrix —
80% of every input tensor was zero padding.** KWS literature uses 25 ms windows with a
10 ms hop (~100 frames/s).

[collect_data_torch.py](collect_data_torch.py) adds a torchaudio front end at standard
framing (25 ms / 10 ms / 40 mel / 13 MFCC + delta, 101 frames). Both feature sets coexist
via `Config.use_feature_set()` and are selected with `--features`.

| Front end | Frames/clip | Non-zero frames | Feature tensor |
|---|---|---|---|
| analyzer.exe | 40 (capped) | 7.88 | (26, 40) |
| torchaudio | 101 | 99.4 | (26, 101) |

### Ablation — same model, same split, same hyper-parameters

| Model | analyzer features | torchaudio features | Δ |
|---|---|---|---|
| DS-CNN (80.7K) | 74.27% | **95.49%** | **+21.2** |
| 1D-ResNet (1.90M) | 74.75% | **96.12%** | **+21.4** |

The first epoch on torchaudio features (87.38%) already beat 40 epochs on analyzer
features (76.48% val). The feature front end, not model capacity, was the bottleneck.

```bash
python collect_data_torch.py --workers 4
python train.py --model dscnn --features torch --speaker-split --epochs 40 --label-smoothing 0.1
```

---

## 3. Keyword spotting — results

30 classes, speaker-disjoint split, test set 6,211 samples
(random baseline 3.33%, majority class 3.49%). Latency: batch=1, single CPU thread.

| Model | Features | Top-1 | Top-3 | Macro-F1 | Params | Latency p50 |
|---|---|---|---|---|---|---|
| 1D-ResNet | torchaudio | **96.12%** | 98.04% | 96.23% | 1.90M | 2.39 ms |
| DS-CNN | torchaudio | **95.49%** | 98.10% | 95.64% | **80.7K** | 1.16 ms |
| 1D-ResNet | analyzer | 74.75% | 89.41% | 75.23% | 1.90M | 1.69 ms |
| DS-CNN | analyzer | 74.27% | 89.15% | 74.78% | 80.7K | 1.09 ms |
| MLP | analyzer | 64.82% | 83.84% | 65.08% | 133K | 0.05 ms |

**Parameter efficiency.** `AudioDSCNN` ([models.py](models.py)) replaces standard
convolutions with depthwise + pointwise pairs: **23.5× fewer parameters for −0.63 points**.

**MLP.** The training path averaged the 40 frames into a single vector, leaving the dense
model 26 inputs and no temporal information (12.36% accuracy). Flattening instead
(`flatten_features()`) brings it to 64.82% — a meaningful no-convolution baseline, ~10
points below the conv models.

**Time-shift augmentation — no gain (measured).** Feature-domain time shift (±100 ms,
scaled per feature set) was added alongside SpecAugment and changed nothing:

| DS-CNN, torchaudio features | Val | Test |
|---|---|---|
| SpecAugment only | 96.29% | **95.49%** |
| SpecAugment + time shift | 96.27% | 95.38% |

−0.11 points, within run-to-run noise. Shifting an MFCC matrix pads with zeros rather
than with real audio, and the global average pooling already gives the models substantial
shift invariance. The ~1 point the literature attributes to augmentation comes from
waveform-domain shift plus background-noise mixing, which requires re-extracting features
each epoch — not implemented here.

### Published comparison (30/35-class settings)

| System | Accuracy | Params | Source |
|---|---|---|---|
| Google CNN baseline (Warden 2018) | 85.4% | — | [arXiv:1804.03209](https://arxiv.org/abs/1804.03209) |
| ResNet-15 (res15) | 95.8% | 238K | [arXiv:2004.08531](https://arxiv.org/abs/2004.08531) |
| **This repo — DS-CNN** | **95.49%** | **80.7K** | — |
| **This repo — 1D-ResNet** | **96.12%** | 1.90M | — |
| MatchboxNet-3x1x64 (v1, 30-class) | 97.21% | 77K | [arXiv:2004.08531](https://arxiv.org/abs/2004.08531) |
| MatchboxNet-3x2x64 (v1, 30-class) | 97.48% | 93K | same |
| Keyword Transformer KWT-3 (v2, 35-class) | ~97.5% | — | [arXiv:2104.00769](https://arxiv.org/abs/2104.00769) |

DS-CNN lands between the res15 baseline and MatchboxNet at comparable size. The remaining
~1.7 points to MatchboxNet are attributable to its residual/dilated blocks, time-shift and
noise augmentation, and longer training schedules — none of which are applied here.

### Remaining error modes (DS-CNN, torchaudio features)

| Pair | Errors |
|---|---|
| three ↔ tree | 21 |
| down ↔ no | 12 |
| off ↔ up | 10 |
| seven ↔ zero | 9 |

Dominated by genuinely near-homophonous pairs — the expected residual error profile.

```bash
python benchmark.py --features torch --speaker-split \
    --weights resnet=audio_resnet_30class_torch.pth dscnn=audio_dscnn_30class_torch.pth
```

---

## 4. Bandwidth extension (8 kHz → 16 kHz)

`AudioUNet.forward()` did not add the input to the output despite documenting a residual
design, so the network had to reconstruct the entire waveform from scratch. Measured
against the naive-upsampling baseline (its own input), it was **20 dB worse than doing
nothing**. Fixed in [models_bwe.py](bandwidth_extension/models_bwe.py).

Final results, 300 test segments (150 s of audio), identical metric for every row:

| System | SNR | LSD |
|---|---|---|
| Naive upsample (model input, i.e. "do nothing") | **29.45 dB** | 23.85 dB |
| AudioUNet, before residual fix | 6.44 dB | 16.67 dB |
| AudioUNet, after residual fix | 27.21 dB | **8.94 dB** |

19.65M parameters, real-time factor 0.032 on 4 CPU threads (31.5× real time).

**Reading these numbers honestly.** The trained model improves LSD by 14.91 dB but is
2.24 dB *below* the naive baseline in SNR. Both directions are real and consistent:
the network adds genuine high-band content (large spectral-envelope gain), while any
added high-frequency energy that is not phase-aligned with the target is penalised by
waveform SNR. On Speech Commands this makes SNR a weak metric — the 4–8 kHz band carries
little energy in these recordings, so simply discarding it already scores 29.45 dB.
As it stands the model is a clear win on spectral fidelity and a small loss on waveform
fidelity; it is not yet unambiguously better than doing nothing.

**On comparing with the literature.** AudioUNet (Kuleshov et al., ICLR 2017 workshop)
reports SNR 21.1 dB / LSD 3.2 dB for 2× VCTK single-speaker, and a spline baseline of
20.3 dB / 4.5 dB. Those numbers are **not** directly comparable to the table above: VCTK is
clean, studio-recorded, single/multi-speaker read speech with seconds of context, while
Speech Commands clips are 1 s crowdsourced words from hundreds of speakers with variable
noise, and their numerically silent high band inflates LSD regardless of model quality.
The honest reference point is the first row — the naive-upsampling baseline measured on
the same data with the same metric.

```bash
python train_bwe.py --limit 3000 --epochs 20 --batch-size 16
python benchmark_bwe.py --model audio_unet_bwe.pth --limit 600
```

---

## 5. Open items

- BWE training is unstable (val SNR swung 18–27.6 dB across epochs at constant LR — the
  plateau scheduler never triggered in 20 epochs). A lower LR or warmup is the first thing
  to try, together with training on more than 3,000 files.
- BWE SNR is still below the naive-upsampling baseline; closing that gap is the concrete
  next objective, not more epochs at the current settings.
- BWE model is 19.65M parameters for a 2× task — likely heavily over-parameterised.
- Waveform-domain augmentation (time shift + background-noise mixing) is still missing.
  The feature-domain time shift tried here gave no gain (§3), so this needs an on-the-fly
  waveform dataset rather than another pass over the cached features.
- `analyzer.exe` remains the default front end; `--features torch` must be passed
  explicitly.
