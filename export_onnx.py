import os
import sys
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import torch
import numpy as np
from models import create_model, load_model_weights
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument("--model", default="resnet", choices=["resnet", "cnn"])
    parser.add_argument("--weights", default="audio_resnet_30class.pth")
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-classes", type=int, default=30)
    parser.add_argument("--in-channels", type=int, default=26)
    parser.add_argument("--num-frames", type=int, default=40)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.model}.onnx"

    Config.init_env()

    print(f"Loading {args.weights}...")
    model = create_model(args.model, num_classes=args.num_classes,
                         in_channels=args.in_channels)
    model = load_model_weights(model, args.weights)
    model.eval()

    dummy_input = torch.randn(1, args.in_channels, args.num_frames)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["mfcc_features"],
        output_names=["logits"],
        dynamic_axes={
            "mfcc_features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,
    )

    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
