"""
Unified training entry point. Supports ResNet, CNN, and MLP.

Examples:
    python train.py --model resnet --epochs 60
    python train.py --model cnn --epochs 80 --batch-size 64 --delta-delta
    python train.py --model resnet --class-weights --label-smoothing 0.1
    python train.py --model resnet --resume checkpoints/epoch_0020.pth
    python train.py --model resnet --scheduler cosine
"""
import argparse
from data_utils import split_and_scale
from models import create_model
from trainer import Trainer, build_loaders, _compute_class_weights
from config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Train audio keyword recognition models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="resnet", choices=["resnet", "cnn", "mlp"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--delta-delta", action="store_true")
    parser.add_argument("--class-weights", action="store_true",
                        help="Use inverse-frequency class weights for imbalance")
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--scheduler", choices=["plateau", "cosine"], default=None)
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--output", default=None, help="Output weights filename")
    args = parser.parse_args()

    # Apply CLI overrides to Config
    if args.lr is not None:
        Config.LR = args.lr
    if args.epochs is not None:
        Config.EPOCHS = args.epochs

    print("=" * 55)
    print(f"  {args.model.upper()} Training")
    print("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test, label_names, _ = \
        split_and_scale(use_delta_delta=args.delta_delta)

    in_channels = X_train.shape[1]
    batch_size = args.batch_size or (64 if args.model == "cnn" else Config.BATCH_SIZE)
    train_loader, val_loader, test_loader = build_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, augment=not args.no_augment,
    )

    if args.model == "mlp":
        X_train_flat = X_train.mean(axis=-1)
        X_val_flat = X_val.mean(axis=-1)
        X_test_flat = X_test.mean(axis=-1)
        train_loader, val_loader, test_loader = build_loaders(
            X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test,
            batch_size=batch_size, augment=False,
        )
        in_channels = X_train_flat.shape[1]

    if args.model == "mlp":
        model = create_model(args.model, num_classes=len(label_names),
                             input_size=in_channels)
    else:
        model = create_model(args.model, num_classes=len(label_names),
                             in_channels=in_channels)

    cw = _compute_class_weights(y_train) if args.class_weights else None
    trainer = Trainer(
        model,
        class_weights=cw,
        label_smoothing=args.label_smoothing,
        scheduler_type=args.scheduler,
    )

    best_acc = trainer.fit(train_loader, val_loader,
                           epochs=Config.EPOCHS, resume=args.resume)

    acc, report = trainer.evaluate(test_loader, label_names=label_names)
    print(f"\n--- Best Val Acc: {best_acc*100:.2f}% | Test Acc: {acc*100:.2f}% ---")
    print(report)

    output = args.output or f"audio_{args.model}_{len(label_names)}class.pth"
    trainer.save_model(output)


if __name__ == "__main__":
    main()
