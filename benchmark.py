"""Benchmark trained keyword-spotting models on the held-out test set.

Reports accuracy, model size and CPU inference latency so results can be
compared against published KWS numbers (see BENCHMARK.md).

    python benchmark.py --weights resnet=audio_resnet_30class.pth cnn=audio_cnn_30class.pth
"""
import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from config import Config
from data_utils import split_and_scale, flatten_features
from models import create_model, load_model_weights
from trainer import build_loaders


def measure_latency(model, sample, runs=200, warmup=20):
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))


def measure_throughput(model, batch, runs=20):
    with torch.no_grad():
        for _ in range(3):
            model(batch)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(batch)
        elapsed = time.perf_counter() - t0
    return batch.size(0) * runs / elapsed


@torch.no_grad()
def evaluate(model, loader):
    all_probs, all_labels = [], []
    for batch_x, batch_y in loader:
        probs = torch.softmax(model(batch_x), dim=1).numpy()
        all_probs.append(probs)
        all_labels.append(batch_y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)

    top3 = np.mean([labels[i] in np.argsort(probs[i])[-3:] for i in range(len(labels))])
    return {
        "top1": float(np.mean(preds == labels)),
        "top3": float(top3),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark KWS models on the test set")
    parser.add_argument("--weights", nargs="+", default=[
        "resnet=audio_resnet_30class.pth",
        "cnn=audio_cnn_30class.pth",
    ], help="model=path pairs")
    parser.add_argument("--delta-delta", action="store_true")
    parser.add_argument("--speaker-split", action="store_true",
                        help="Speaker-disjoint split (official Speech Commands protocol)")
    parser.add_argument("--features", default=None, choices=["analyzer", "torch"],
                        help="Feature set the weights were trained on")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch CPU threads used for latency measurement")
    parser.add_argument("--output", default="BENCHMARK_RESULTS.md")
    args = parser.parse_args()

    Config.init_env()
    if args.features is not None:
        Config.use_feature_set(args.features)
    torch.set_num_threads(args.threads)

    X_train, X_val, X_test, y_train, y_val, y_test, label_names, _ = \
        split_and_scale(use_delta_delta=args.delta_delta,
                        speaker_split=args.speaker_split)
    loaders = {}
    rows = []
    for spec in args.weights:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        # Dense models consume flattened features; conv models keep (C, T).
        if name == "mlp":
            feats = tuple(flatten_features(X) for X in (X_train, X_val, X_test))
        else:
            feats = (X_train, X_val, X_test)

        if name not in loaders:
            *_, loaders[name] = build_loaders(feats[0], y_train, feats[1], y_val,
                                              feats[2], y_test)
        test_loader = loaders[name]
        single = torch.FloatTensor(feats[2][:1])
        batch = torch.FloatTensor(feats[2][:128])

        if name == "mlp":
            model = create_model(name, num_classes=len(label_names),
                                 input_size=feats[0].shape[1])
        else:
            model = create_model(name, num_classes=len(label_names),
                                 in_channels=X_train.shape[1])
        model = load_model_weights(model, path)
        model.eval()

        metrics = evaluate(model, test_loader)
        p50, p95 = measure_latency(model, single)
        rows.append({
            "name": name,
            "path": path,
            "params": sum(p.numel() for p in model.parameters()),
            "size_mb": os.path.getsize(path) / 1e6,
            "latency_p50": p50,
            "latency_p95": p95,
            "throughput": measure_throughput(model, batch),
            **metrics,
        })
        print(f"{name:8s} top1={metrics['top1']*100:.2f}% top3={metrics['top3']*100:.2f}% "
              f"f1={metrics['macro_f1']*100:.2f}% p50={p50:.2f}ms")

    majority = float(np.max(np.bincount(y_test)) / len(y_test))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# KWS Benchmark Results\n\n")
        split_name = "speaker-disjoint (official protocol)" if args.speaker_split else "random per-utterance"
        f.write(f"Split: {split_name}\n")
        f.write(f"Test set: {len(y_test)} samples, {len(label_names)} classes "
                f"(majority-class baseline {majority*100:.2f}%, random {100/len(label_names):.2f}%)\n")
        f.write(f"CPU threads: {args.threads}, batch=1 latency, batch=128 throughput\n\n")
        f.write("| Model | Top-1 | Top-3 | Macro-F1 | Params | Size | Latency p50 / p95 | Throughput |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['top1']*100:.2f}% | {r['top3']*100:.2f}% | "
                    f"{r['macro_f1']*100:.2f}% | {r['params']/1e3:.0f}K | {r['size_mb']:.1f} MB | "
                    f"{r['latency_p50']:.2f} / {r['latency_p95']:.2f} ms | "
                    f"{r['throughput']:.0f} samples/s |\n")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
