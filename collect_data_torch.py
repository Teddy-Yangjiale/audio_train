"""Extract MFCC features with torchaudio at standard KWS framing.

analyzer.exe emits ~8 frames per 1s clip (125 ms hop), so 80% of every 40-frame
feature matrix is zero padding. This extractor uses the 25 ms window / 10 ms hop
framing used by the KWS literature, producing ~101 frames per clip.

    python collect_data_torch.py                 # all 30 classes
    python collect_data_torch.py --workers 4     # limit CPU usage
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torchaudio

from config import Config

SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 40
N_FFT = 512
WIN_LENGTH = 400        # 25 ms
HOP_LENGTH = 160        # 10 ms
NUM_FRAMES = 101        # 1 s of audio
SAMPLES_PER_CLASS = 2000

_mfcc_transform = None


def get_transform():
    global _mfcc_transform
    if _mfcc_transform is None:
        _mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                "n_fft": N_FFT,
                "n_mels": N_MELS,
                "win_length": WIN_LENGTH,
                "hop_length": HOP_LENGTH,
                "center": True,
            },
        )
    return _mfcc_transform


def process_single_file(task):
    file_path, label_idx = task
    try:
        sr, data = wavfile.read(file_path)
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        else:
            audio = data.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            return None

        mfcc = get_transform()(torch.from_numpy(audio))          # (13, frames)
        delta = torchaudio.functional.compute_deltas(mfcc)
        feat = torch.cat([mfcc, delta], dim=0).T.numpy()         # (frames, 26)

        if len(feat) < NUM_FRAMES:
            feat = np.pad(feat, ((0, NUM_FRAMES - len(feat)), (0, 0)))
        else:
            feat = feat[:NUM_FRAMES]

        return feat.astype(np.float32), label_idx, file_path
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract MFCC features via torchaudio")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 4))
    parser.add_argument("--limit", type=int, default=SAMPLES_PER_CLASS,
                        help="max files per class")
    args = parser.parse_args()

    torch.set_num_threads(1)  # workers parallelise, keep each single-threaded

    labels = sorted([d for d in os.listdir(Config.DATA_DIR)
                     if os.path.isdir(os.path.join(Config.DATA_DIR, d))
                     and d not in Config.EXCLUDE])
    print(f"Found {len(labels)} classes")

    tasks = []
    for idx, label in enumerate(labels):
        folder = os.path.join(Config.DATA_DIR, label)
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        tasks.extend((os.path.join(folder, f), idx) for f in files[:args.limit])

    print(f"Extracting {len(tasks)} files with {args.workers} workers "
          f"({WIN_LENGTH/SAMPLE_RATE*1000:.0f} ms window, "
          f"{HOP_LENGTH/SAMPLE_RATE*1000:.0f} ms hop, {NUM_FRAMES} frames)")

    with Pool(args.workers) as pool:
        results = pool.map(process_single_file, tasks, chunksize=64)

    features, label_ids, paths = [], [], []
    for res in results:
        if res is not None:
            features.append(res[0])
            label_ids.append(res[1])
            paths.append(res[2])

    X = np.array(features)
    np.save("X_features_torch.npy", X)
    np.save("y_labels_torch.npy", np.array(label_ids))
    np.save("label_map_torch.npy", np.array(labels))
    np.save("paths_torch.npy", np.array(paths))

    nonzero = (np.abs(X).sum(axis=2) > 0).sum(axis=1)
    print(f"Done. {len(features)} samples, shape {X.shape}")
    print(f"Non-zero frames per sample: mean {nonzero.mean():.1f} / {NUM_FRAMES}")


if __name__ == "__main__":
    main()
