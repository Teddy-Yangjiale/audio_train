import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_raw_features


def main():
    X, y, label_names = load_raw_features()
    num_samples = X.shape[0]
    num_frames = X.shape[1]
    num_features = X.shape[2]
    num_classes = len(label_names)

    counts = np.bincount(y, minlength=num_classes)

    print("=" * 55)
    print("  Dataset Statistics")
    print("=" * 55)
    print(f"  Samples:     {num_samples}")
    print(f"  Classes:     {num_classes}")
    print(f"  Time frames: {num_frames}")
    print(f"  Features:    {num_features} (13 MFCC + 13 Delta)")
    print(f"  Samples/class: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.0f}, median={np.median(counts):.0f}")
    print()

    # Per-class breakdown
    header = f"  {'Class':<10s} {'Count':>6s} {'%':>7s}  {'Imbalance':>10s}"
    print(header)
    print("  " + "-" * 38)
    for i, name in enumerate(label_names):
        pct = counts[i] / num_samples * 100
        ratio = counts[i] / counts.mean()
        bar = "#" * int(ratio * 10)
        print(f"  {name:<10s} {counts[i]:>6d} {pct:>6.1f}%  {bar}")

    print()
    print(f"  Value range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")

    # Save plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(num_classes), counts)
    ax1.set_title("Samples per Class")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.axhline(counts.mean(), color="red", linestyle="--", label=f"Mean ({counts.mean():.0f})")
    ax1.legend()

    # Mean feature across time for a few classes
    for idx in range(min(5, num_classes)):
        mask = y == idx
        mean_feat = X[mask].mean(axis=(0, 1))
        ax2.plot(mean_feat, label=label_names[idx])
    ax2.set_title("Mean Feature Values (first 5 classes)")
    ax2.set_xlabel("Feature index")
    ax2.set_ylabel("Mean value")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("dataset_stats.png", dpi=150)
    print("\nPlot saved to dataset_stats.png")
    plt.show()


if __name__ == "__main__":
    main()
