import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_utils import split_and_scale
from models import create_model, load_model_weights
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Visualize feature embeddings via t-SNE")
    parser.add_argument("--model", default="resnet", choices=["resnet", "cnn"])
    parser.add_argument("--weights", default="audio_resnet_30class.pth")
    parser.add_argument("--delta-delta", action="store_true")
    parser.add_argument("--samples", type=int, default=3000, help="Max samples for t-SNE")
    parser.add_argument("--output", default="tsne_embeddings.png")
    args = parser.parse_args()

    Config.init_env()

    print("Loading data and model...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_names, _ = \
        split_and_scale(use_delta_delta=args.delta_delta)

    model = create_model(args.model, num_classes=len(label_names),
                         in_channels=X_train.shape[1])
    model = load_model_weights(model, args.weights)
    model.eval()

    X_subset = X_test[:args.samples]
    y_subset = y_test[:args.samples]

    model_cpu = model
    extractor = torch.nn.Sequential(*list(model_cpu.children())[:-1])

    with torch.no_grad():
        features = extractor(torch.FloatTensor(X_subset))
        features = features.view(features.size(0), -1).numpy()

    print(f"Running t-SNE on {len(features)} samples, {features.shape[1]}-dim features...")
    tsne = TSNE(n_components=2, random_state=Config.RANDOM_SEED, perplexity=30, n_iter=1000)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=y_subset, cmap='tab20',
                          alpha=0.6, s=8)
    cbar = plt.colorbar(scatter, ticks=range(len(label_names)))
    cbar.ax.set_yticklabels(label_names)
    cbar.set_label('Class')
    plt.title(f't-SNE Visualization of {args.model.upper()} Embeddings')
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
