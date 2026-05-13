import os
import sys
import subprocess
import json
import numpy as np
from tqdm import tqdm
from config import Config

# Optional VAD support
try:
    import webrtcvad
    import struct
    import wave
    HAS_VAD = True
except ImportError:
    HAS_VAD = False

DATASET_PATH = Config.DATA_DIR
ANALYZER_PATH = Config.ANALYZER_PATH
EXCLUDE = Config.EXCLUDE
MAX_LEN = Config.NUM_FRAMES
SAMPLES_PER_CLASS = 2000

USE_VAD = "--vad" in sys.argv


def read_wav_frames(file_path, vad, aggressiveness=2):
    """Trim leading/trailing silence using webrtcvad."""
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        wf.close()
        return None
    frames_data = wf.readframes(wf.getnframes())
    wf.close()

    frame_ms = 30
    frame_size = int(16000 * frame_ms / 1000)  # 480 samples
    n = len(frames_data) // (frame_size * 2)

    voice_flags = []
    for i in range(n):
        chunk = frames_data[i * frame_size * 2:(i + 1) * frame_size * 2]
        try:
            is_speech = vad.is_speech(chunk, 16000)
            voice_flags.append(is_speech)
        except Exception:
            voice_flags.append(True)

    if not any(voice_flags):
        return None
    first = voice_flags.index(True)
    last = len(voice_flags) - 1 - voice_flags[::-1].index(True)

    start_sample = first * frame_size
    end_sample = (last + 1) * frame_size
    return frames_data[start_sample * 2:end_sample * 2], 16000


def extract_mfcc(file_path):
    result = subprocess.check_output([ANALYZER_PATH, file_path], stderr=subprocess.DEVNULL)
    data = json.loads(result)
    return np.array([f["mfcc"] for f in data])


def pad_or_truncate(mfccs):
    if len(mfccs) < MAX_LEN:
        mfccs = np.pad(mfccs, ((0, MAX_LEN - len(mfccs)), (0, 0)), mode="constant")
    else:
        mfccs = mfccs[:MAX_LEN, :]
    return mfccs


def build_feat_matrix(mfccs):
    delta = np.diff(mfccs, axis=0, prepend=mfccs[:1])
    return np.hstack([mfccs, delta])


def main():
    if USE_VAD and HAS_VAD:
        print("VAD trimming enabled (webrtcvad)")
    elif USE_VAD and not HAS_VAD:
        print("Warning: webrtcvad not installed, VAD disabled. Install with: pip install webrtcvad")
    else:
        print("VAD disabled (use --vad to enable)")

    TARGET_LABELS = sorted([d for d in os.listdir(DATASET_PATH)
                           if os.path.isdir(os.path.join(DATASET_PATH, d)) and d not in EXCLUDE])

    print(f"Found {len(TARGET_LABELS)} classes.")
    print(f"Classes: {', '.join(TARGET_LABELS)}")

    features, labels = [], []

    for label_idx, label in enumerate(TARGET_LABELS):
        folder = os.path.join(DATASET_PATH, label)
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        files_to_process = files[:SAMPLES_PER_CLASS]

        vad = webrtcvad.Vad(2) if (USE_VAD and HAS_VAD) else None

        for file_name in tqdm(files_to_process, desc=f"Processing: {label}"):
            file_path = os.path.join(folder, file_name)
            try:
                mfccs = extract_mfcc(file_path)
                mfccs = pad_or_truncate(mfccs)
                feat_matrix = build_feat_matrix(mfccs)
                features.append(feat_matrix)
                labels.append(label_idx)
            except Exception:
                continue

    np.save(Config.FEATURE_FILE, np.array(features))
    np.save(Config.LABEL_FILE, np.array(labels))
    np.save(Config.LABEL_MAP_FILE, np.array(TARGET_LABELS))

    print(f"Done. Total samples: {len(features)} | Classes: {len(TARGET_LABELS)}")


if __name__ == "__main__":
    main()
