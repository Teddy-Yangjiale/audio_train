"""
Train AudioUNet for bandwidth extension (low SR -> high SR).

Examples:
    python train_bwe.py --epochs 50
    python train_bwe.py --epochs 100 --batch-size 16
    python train_bwe.py --data-dir D:/datasets/vctk --low-sr 8000 --high-sr 22050
"""
import argparse
import torch
from config_bwe import BWEConfig
from models_bwe import AudioUNet
from data_utils_bwe import build_dataloaders
from trainer_bwe import BWETrainer


def main():
    parser = argparse.ArgumentParser(description="Train AudioUNet for Bandwidth Extension")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--low-sr", type=int, default=None)
    parser.add_argument("--high-sr", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of WAV files (for quick testing)")
    parser.add_argument("--output", default="audio_unet_bwe.pth")
    args = parser.parse_args()

    if args.low_sr is not None:
        BWEConfig.LOW_SR = args.low_sr
    if args.high_sr is not None:
        BWEConfig.HIGH_SR = args.high_sr
        BWEConfig.SEGMENT_SAMPLES = int(BWEConfig.HIGH_SR * BWEConfig.SEGMENT_SEC)
    if args.data_dir is not None:
        BWEConfig.DATA_DIR = args.data_dir
    if args.lr is not None:
        BWEConfig.LR = args.lr
    if args.batch_size is not None:
        BWEConfig.BATCH_SIZE = args.batch_size

    epochs = args.epochs or BWEConfig.EPOCHS

    print("=" * 60)
    print("  Audio Bandwidth Extension -> AudioUNet Training")
    print(f"  Low SR: {BWEConfig.LOW_SR} Hz -> High SR: {BWEConfig.HIGH_SR} Hz")
    print(f"  Segment: {BWEConfig.SEGMENT_SAMPLES / BWEConfig.HIGH_SR:.1f}s "
          f"({BWEConfig.SEGMENT_SAMPLES} samples)")
    print("=" * 60)

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size, file_limit=args.limit)

    model = AudioUNet(
        base_channels=BWEConfig.BASE_CHANNELS,
        num_encoders=BWEConfig.NUM_ENCODERS,
        kernel_size=BWEConfig.KERNEL_SIZE,
        stride=BWEConfig.STRIDE,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AudioUNet parameters: {n_params:,}")

    trainer = BWETrainer(model)
    trainer.fit(train_loader, val_loader, epochs=epochs)

    metrics, snrs, lsds = trainer.evaluate(test_loader)
    print(f"\n--- Test Results ---")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"  SNR (dB):  {metrics['snr_db']:.2f}")
    print(f"  LSD:       {metrics['lsd']:.4f}")

    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
