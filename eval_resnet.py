import argparse
import torch
import numpy as np
from data_utils import split_and_scale, load_scaler
from models import create_model, load_model_weights
from trainer import build_loaders
from config import Config
from sklearn.metrics import classification_report, confusion_matrix


def top_k_accuracy(y_true, y_pred_probs, k=3):
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])


def most_confused_pairs(cm, label_names, top_n=10):
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(len(label_names)):
        for j in range(i + 1, len(label_names)):
            score = cm[i, j] + cm[j, i]
            if score > 0:
                pairs.append((label_names[i], label_names[j], int(score)))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    parser.add_argument("--model", default="resnet", choices=["resnet", "cnn", "dscnn"])
    parser.add_argument("--weights", default="audio_resnet_30class.pth")
    parser.add_argument("--delta-delta", action="store_true")
    parser.add_argument("--speaker-split", action="store_true",
                        help="Speaker-disjoint split (official Speech Commands protocol)")
    parser.add_argument("--features", default=None, choices=["analyzer", "torch"])
    args = parser.parse_args()

    Config.init_env()
    if args.features is not None:
        Config.use_feature_set(args.features)

    print("=" * 55)
    print(f"  Evaluate {args.model.upper()} on Test Set")
    print("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test, label_names, _ = \
        split_and_scale(use_delta_delta=args.delta_delta,
                        speaker_split=args.speaker_split)
    _, _, test_loader = build_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

    model = create_model(args.model, num_classes=len(label_names),
                         in_channels=X_train.shape[1])
    model = load_model_weights(model, args.weights)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    top3 = top_k_accuracy(all_labels, all_probs, k=3)

    report = classification_report(all_labels, all_preds, target_names=label_names)

    print(report)
    print(f"Top-1 Accuracy: {acc*100:.2f}%")
    print(f"Top-3 Accuracy: {top3*100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    pairs = most_confused_pairs(cm, label_names)
    print("\n--- Most Confused Word Pairs ---")
    for a, b, count in pairs:
        print(f"  {a:<8s} <-> {b:<8s}  ({count} errors)")

    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{args.model.upper()} Evaluation Report\n{'='*55}\n\n")
        f.write(report)
        f.write(f"\nTop-1 Accuracy: {acc*100:.2f}%\n")
        f.write(f"Top-3 Accuracy: {top3*100:.2f}%\n\n")
        f.write("Most Confused Pairs:\n")
        for a, b, count in pairs:
            f.write(f"  {a:<8s} <-> {b:<8s}  ({count} errors)\n")

    print("\nReport saved to evaluation_report.txt")


if __name__ == "__main__":
    main()
