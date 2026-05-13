import argparse
from data_utils import split_and_scale
from models import create_model
from trainer import Trainer, build_loaders
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Train AudioResNet")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--delta-delta", action="store_true", help="Use delta-delta features")
    args = parser.parse_args()

    print("=" * 55)
    print("  AudioResNet Training")
    print("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test, label_names, _ = \
        split_and_scale(use_delta_delta=args.delta_delta)
    train_loader, val_loader, test_loader = build_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=args.batch_size, augment=not args.no_augment,
    )

    in_channels = X_train.shape[1]
    model = create_model("resnet", num_classes=len(label_names), in_channels=in_channels)
    trainer = Trainer(model)

    best_acc = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    acc, report = trainer.evaluate(test_loader, label_names=label_names)
    print(f"\n--- Final Test Accuracy: {acc*100:.2f}% ---")
    print(report)

    trainer.save_model("audio_resnet_30class.pth")


if __name__ == "__main__":
    main()
