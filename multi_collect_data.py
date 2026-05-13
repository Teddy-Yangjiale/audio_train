import os
import sys
import subprocess
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from config import Config

DATASET_PATH = Config.DATA_DIR
ANALYZER_PATH = Config.ANALYZER_PATH
EXCLUDE = Config.EXCLUDE
MAX_LEN = Config.NUM_FRAMES
SAMPLES_PER_CLASS = 2000


def process_single_file(args):
    file_path, label_idx = args
    try:
        result = subprocess.check_output([ANALYZER_PATH, file_path], stderr=subprocess.DEVNULL)
        data = json.loads(result)
        mfccs = np.array([f["mfcc"] for f in data])

        if len(mfccs) < MAX_LEN:
            mfccs = np.pad(mfccs, ((0, MAX_LEN - len(mfccs)), (0, 0)), mode="constant")
        else:
            mfccs = mfccs[:MAX_LEN, :]

        delta = np.diff(mfccs, axis=0, prepend=mfccs[:1])
        feat_matrix = np.hstack([mfccs, delta])

        return feat_matrix, label_idx
    except Exception:
        return None


if __name__ == "__main__":
    TARGET_LABELS = sorted([d for d in os.listdir(DATASET_PATH)
                           if os.path.isdir(os.path.join(DATASET_PATH, d)) and d not in EXCLUDE])

    print(f"Found {len(TARGET_LABELS)} classes.")

    task_list = []
    for label_idx, label in enumerate(TARGET_LABELS):
        folder = os.path.join(DATASET_PATH, label)
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        for file_name in files[:SAMPLES_PER_CLASS]:
            task_list.append((os.path.join(folder, file_name), label_idx))

    print(f"Total tasks: {len(task_list)}")

    num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} CPU cores for parallel processing...")

    features, labels = [], []

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, task_list),
                            total=len(task_list), desc="Parallel collect"))

    print("Aggregating results...")
    for res in results:
        if res is not None:
            features.append(res[0])
            labels.append(res[1])

    np.save(Config.FEATURE_FILE, np.array(features))
    np.save(Config.LABEL_FILE, np.array(labels))
    np.save(Config.LABEL_MAP_FILE, np.array(TARGET_LABELS))

    print(f"Done. Total samples: {len(features)} | Classes: {len(TARGET_LABELS)}")
