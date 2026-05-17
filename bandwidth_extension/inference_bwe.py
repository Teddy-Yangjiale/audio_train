"""
Inference: upsample a low-sample-rate WAV to high-sample-rate.
    python inference_bwe.py input.wav output.wav --model audio_unet_bwe.pth
"""
import argparse
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
from config_bwe import BWEConfig
from models_bwe import AudioUNet


def main():
    parser = argparse.ArgumentParser(description="Upsample audio via AudioUNet")
    parser.add_argument("input", help="Input low-sample-rate WAV file")
    parser.add_argument("output", help="Output high-sample-rate WAV file")
    parser.add_argument("--model", default="audio_unet_bwe.pth", help="Trained model weights")
    parser.add_argument("--low-sr", type=int, default=8000)
    parser.add_argument("--high-sr", type=int, default=16000)
    args = parser.parse_args()

    BWEConfig.LOW_SR = args.low_sr
    BWEConfig.HIGH_SR = args.high_sr
    BWEConfig.init_env()

    print(f"Loading {args.input}...")
    sr, data = wavfile.read(args.input)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample input to target low SR if needed
    if sr != args.low_sr:
        lo_len = int(len(data) * args.low_sr / sr)
        data = signal.resample(data, lo_len)
        sr = args.low_sr

    print("Upsampling to target SR (naive interpolation)...")
    hi_len = int(len(data) * args.high_sr / args.low_sr)
    up = signal.resample(data, hi_len)
    up_tensor = torch.from_numpy(up.astype(np.float32)).unsqueeze(0)

    print("Loading AudioUNet...")
    model = AudioUNet(
        base_channels=BWEConfig.BASE_CHANNELS,
        num_encoders=BWEConfig.NUM_ENCODERS,
        kernel_size=BWEConfig.KERNEL_SIZE,
        stride=BWEConfig.STRIDE,
    )
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    print("Refining with AudioUNet...")
    refined = model.infer(up_tensor)

    output = refined.squeeze(0).numpy()
    output = np.clip(output, -1, 1)
    output_int16 = (output * 32767).astype(np.int16)
    wavfile.write(args.output, args.high_sr, output_int16)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
