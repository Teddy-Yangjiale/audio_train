import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_utils import split_and_scale
from models import create_model, load_model_weights
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix visualization")
    parser.add_argument("--model", default="resnet", choices=["resnet", "cnn"])
    parser.add_argument("--weights", default="audio_resnet_30class.pth")
    parser.add_argument("--delta-delta", action="store_true")
    parser.add_argument("--output", default="confusion_matrix.png")
    args = parser.parse_args()

    Config.init_env()

    print("Loading data and model...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_names, _ = \
        split_and_scale(use_delta_delta=args.delta_delta)

    model = create_model(args.model, num_classes=len(label_names),
                         in_channels=X_train.shape[1])
    model = load_model_weights(model, args.weights)
    model.eval()

    with torch.no_grad():
        preds = torch.argmax(model(torch.FloatTensor(X_test)), dim=1).numpy()

    cm = confusion_matrix(y_test, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    np.nan_to_num(cm_percent, copy=False)

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm_percent, annot=True, fmt=".0f", cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, linecolor='gray',
                annot_kws={"size": 7})
    plt.title(f'Normalized Confusion Matrix ({args.model.upper()})')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
