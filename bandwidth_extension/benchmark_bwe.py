"""Benchmark AudioUNet against the naive-upsampling baseline on the test split.

The model input is already a naive (sinc-resampled) 8k -> 16k signal, so its
SNR/LSD is the baseline any learned model must beat. Also reports parameter
count and real-time factor.

    python benchmark_bwe.py --model audio_unet_bwe.pth --limit 600
"""
import argparse
import time

import numpy as np
import torch

from config_bwe import BWEConfig
from data_utils_bwe import build_dataloaders
from losses_bwe import compute_snr, compute_lsd
from models_bwe import AudioUNet


def summarize(name, snrs, lsds):
    snrs, lsds = np.array(snrs), np.array(lsds)
    return {
        "name": name,
        "snr_mean": snrs.mean(), "snr_median": np.median(snrs),
        "lsd_mean": lsds.mean(), "lsd_median": np.median(lsds),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark AudioUNet bandwidth extension")
    parser.add_argument("--model", default="audio_unet_bwe.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=600,
                        help="limit WAV files scanned (test split is 10% of these)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="BENCHMARK_BWE_RESULTS.md")
    args = parser.parse_args()

    BWEConfig.init_env()
    torch.set_num_threads(args.threads)

    _, _, test_loader = build_dataloaders(batch_size=args.batch_size,
                                          file_limit=args.limit)

    model = AudioUNet(
        base_channels=BWEConfig.BASE_CHANNELS,
        num_encoders=BWEConfig.NUM_ENCODERS,
        kernel_size=BWEConfig.KERNEL_SIZE,
        stride=BWEConfig.STRIDE,
    )
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    base_snr, base_lsd, model_snr, model_lsd = [], [], [], []
    n_samples, infer_time = 0, 0.0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            t0 = time.perf_counter()
            pred = model(batch_x)
            infer_time += time.perf_counter() - t0
            n_samples += batch_x.size(0)
            for i in range(batch_x.size(0)):
                base_snr.append(compute_snr(batch_x[i], batch_y[i]))
                base_lsd.append(compute_lsd(batch_x[i], batch_y[i]))
                model_snr.append(compute_snr(pred[i], batch_y[i]))
                model_lsd.append(compute_lsd(pred[i], batch_y[i]))

    baseline = summarize("Naive upsample (input)", base_snr, base_lsd)
    learned = summarize("AudioUNet", model_snr, model_lsd)

    audio_sec = n_samples * BWEConfig.SEGMENT_SAMPLES / BWEConfig.HIGH_SR
    rtf = infer_time / audio_sec
    params = sum(p.numel() for p in model.parameters())

    print(f"\nTest segments: {n_samples} ({audio_sec:.1f}s of audio)")
    for r in (baseline, learned):
        print(f"  {r['name']:24s} SNR {r['snr_mean']:6.2f} dB (med {r['snr_median']:6.2f}) | "
              f"LSD {r['lsd_mean']:6.2f} dB (med {r['lsd_median']:6.2f})")
    print(f"  delta: SNR {learned['snr_mean'] - baseline['snr_mean']:+.2f} dB, "
          f"LSD {learned['lsd_mean'] - baseline['lsd_mean']:+.2f} dB")
    print(f"  params {params/1e6:.1f}M, real-time factor {rtf:.3f} "
          f"({1/rtf:.1f}x real time on {args.threads} CPU thread(s))")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Bandwidth Extension Benchmark (8 kHz -> 16 kHz)\n\n")
        f.write(f"Data: Google Speech Commands, {n_samples} test segments "
                f"({BWEConfig.SEGMENT_SEC}s each, {audio_sec:.0f}s total)\n\n")
        f.write("| System | SNR mean | SNR median | LSD mean | LSD median |\n|---|---|---|---|---|\n")
        for r in (baseline, learned):
            f.write(f"| {r['name']} | {r['snr_mean']:.2f} dB | {r['snr_median']:.2f} dB | "
                    f"{r['lsd_mean']:.2f} dB | {r['lsd_median']:.2f} dB |\n")
        f.write(f"\nParameters: {params:,} | real-time factor: {rtf:.3f} "
                f"({args.threads} CPU thread(s))\n")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
