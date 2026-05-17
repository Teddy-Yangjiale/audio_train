import os
import glob
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
from config_bwe import BWEConfig


def find_wav_files(data_dir, limit=None):
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    exclude = "_background_noise_"
    files = [f for f in files if exclude not in f]
    if limit:
        files = files[:limit]
    return files


def load_audio(file_path, target_sr, duration_sec=None):
    """Load a WAV file and resample to target_sr. Returns (T,) float32 tensor."""
    sr, data = wavfile.read(file_path)
    # Convert to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample
    if sr != target_sr:
        numsamples = int(len(data) * target_sr / sr)
        data = signal.resample(data, numsamples)
    # Trim
    if duration_sec is not None:
        max_samples = int(target_sr * duration_sec)
        if len(data) > max_samples:
            start = np.random.randint(0, len(data) - max_samples + 1)
            data = data[start:start + max_samples]
    return torch.from_numpy(data.copy()).float()


def make_low_res(high_res, low_sr=None, high_sr=None):
    """Downsample to low_sr then upsample back to high_sr (naive interpolation)."""
    if low_sr is None:
        low_sr = BWEConfig.LOW_SR
    if high_sr is None:
        high_sr = BWEConfig.HIGH_SR

    audio = high_res.numpy()
    # Down
    lo_len = int(len(audio) * low_sr / high_sr)
    lo = signal.resample(audio, lo_len)
    # Up
    hi_len = int(len(lo) * high_sr / low_sr)
    up = signal.resample(lo, hi_len)
    # Match length
    if len(up) < len(audio):
        up = np.pad(up, (0, len(audio) - len(up)))
    elif len(up) > len(audio):
        up = up[:len(audio)]
    return torch.from_numpy(up.astype(np.float32))


def prepare_dataset(file_paths, segment_samples, split="train"):
    """Process a list of WAV files into fixed-length (input, target) pairs.

    Each pair:
        input:  naive-upsampled low-res version
        target: original high-res audio
    """
    inputs, targets = [], []
    max_per_file = 4 if split == "train" else 1

    for fp in file_paths:
        try:
            audio = load_audio(fp, BWEConfig.HIGH_SR)
        except Exception:
            continue

        if audio.shape[-1] < segment_samples:
            audio = torch.nn.functional.pad(audio, (0, segment_samples - audio.shape[-1]))

        for _ in range(max_per_file):
            max_start = max(0, audio.shape[-1] - segment_samples)
            start = np.random.randint(0, max_start + 1)
            target = audio[start:start + segment_samples]
            inp = make_low_res(target)
            inputs.append(inp.unsqueeze(0))  # (1, T)
            targets.append(target.unsqueeze(0))

    return torch.stack(inputs), torch.stack(targets)


def build_dataloaders(batch_size=None, file_limit=None):
    """Find WAV files, split train/val/test, build DataLoader."""
    if batch_size is None:
        batch_size = BWEConfig.BATCH_SIZE

    files = find_wav_files(BWEConfig.DATA_DIR, limit=file_limit)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No WAV files found under {BWEConfig.DATA_DIR}. "
            "Please set BWEConfig.DATA_DIR to a directory with .wav files."
        )

    np.random.seed(BWEConfig.RANDOM_SEED)
    np.random.shuffle(files)
    n = len(files)
    n_test = int(n * BWEConfig.TEST_SPLIT)
    n_val = int(n * BWEConfig.VAL_SPLIT)

    test_files = files[:n_test]
    val_files = files[n_test:n_test + n_val]
    train_files = files[n_test + n_val:]

    print(f"Files: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    segment_samples = BWEConfig.SEGMENT_SAMPLES

    print("Preparing training data...")
    X_train, y_train = prepare_dataset(train_files, segment_samples, "train")
    print("Preparing validation data...")
    X_val, y_val = prepare_dataset(val_files, segment_samples, "val")
    print("Preparing test data...")
    X_test, y_test = prepare_dataset(test_files, segment_samples, "test")

    from torch.utils.data import DataLoader, TensorDataset

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False,
    )

    print(f"Train segments: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return train_loader, val_loader, test_loader
