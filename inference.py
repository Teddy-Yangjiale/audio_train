import argparse
from config import Config
from data_utils import preprocess_single, load_scaler
from models import create_model, load_model_weights
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Infer a single WAV file")
    parser.add_argument("wav_path", help="Path to WAV file")
    parser.add_argument("--model", default="resnet", choices=["resnet", "cnn"])
    parser.add_argument("--weights", default="audio_resnet_30class.pth")
    parser.add_argument("--label-map", default=Config.LABEL_MAP_FILE)
    args = parser.parse_args()

    Config.init_env()

    label_names = np.load(args.label_map, allow_pickle=True)
    scaler = load_scaler()

    features = preprocess_single(args.wav_path, scaler=scaler)

    model = create_model(args.model, num_classes=len(label_names))
    model = load_model_weights(model, args.weights)
    model.eval()

    with torch.no_grad():
        logits = model(torch.FloatTensor(features))
        probs = torch.softmax(logits, dim=1)[0].numpy()
        top_indices = np.argsort(probs)[::-1][:5]

    print(f"\nFile: {args.wav_path}")
    print("-" * 40)
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {label_names[idx]:<10s}  {probs[idx]*100:.1f}%")

    best_idx = top_indices[0]
    print(f"\nResult: {label_names[best_idx]} ({probs[best_idx]*100:.1f}%)")


if __name__ == "__main__":
    main()
