"""
Evaluate AudioUNet on test set: SNR, LSD, and save comparison spectrograms.
    python evaluate_bwe.py --model audio_unet_bwe.pth
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from config_bwe import BWEConfig
from models_bwe import AudioUNet
from data_utils_bwe import build_dataloaders
from losses_bwe import compute_snr, compute_lsd
from losses_bwe import stft_mag


def main():
    parser = argparse.ArgumentParser(description="Evaluate AudioUNet bandwidth extension")
    parser.add_argument("--model", default="audio_unet_bwe.pth")
    parser.add_argument("--batch-size", type=int, default=BWEConfig.BATCH_SIZE)
    args = parser.parse_args()

    BWEConfig.init_env()

    print("Loading data...")
    _, _, test_loader = build_dataloaders(batch_size=args.batch_size)

    print("Loading model...")
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

    snrs, lsds = [], []
    all_preds, all_targets = [], []

    for batch_x, batch_y in test_loader:
        pred = model(batch_x)
        for i in range(batch_x.size(0)):
            snrs.append(compute_snr(pred[i], batch_y[i]))
            lsds.append(compute_lsd(pred[i], batch_y[i]))
        all_preds.append(pred)
        all_targets.append(batch_y)

    snrs = np.array(snrs)
    lsds = np.array(lsds)

    print(f"\n--- Evaluation Results ---")
    print(f"  SNR:  mean={snrs.mean():.2f} dB, median={np.median(snrs):.2f} dB, "
          f"std={snrs.std():.2f}")
    print(f"  LSD:  mean={lsds.mean():.4f}, median={np.median(lsds):.4f}, "
          f"std={lsds.std():.4f}")

    print("\n--- SNR Distribution ---")
    bins = [-100, -10, 0, 10, 20, 30, 50, 100]
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = ((snrs >= lo) & (snrs < hi)).sum()
        pct = count / len(snrs) * 100
        bar = "#" * int(pct)
        print(f"  [{lo:>4d}, {hi:>3d}): {count:>5d} ({pct:5.1f}%) {bar}")

    # Spectrogram comparison for one sample
    pred0 = all_preds[0][0]
    target0 = all_targets[0][0]
    inp0 = next(iter(test_loader))[0][0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Input (naive upsample)", "AudioUNet", "Ground Truth"]
    for ax, signal, title in zip(axes, [inp0, pred0, target0], titles):
        mag = stft_mag(signal.unsqueeze(0), 2048, 512)
        ax.imshow(20 * torch.log10(mag + 1e-6).numpy(), aspect="auto",
                  origin="lower", cmap="magma")
        ax.set_title(f"{title}\nSNR: {compute_snr(signal, target0):.1f} dB")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Freq bin")

    plt.tight_layout()
    plt.savefig("spectrogram_comparison.png", dpi=200)
    print("Spectrogram comparison saved to spectrogram_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
