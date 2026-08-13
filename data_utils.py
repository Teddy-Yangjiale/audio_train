import hashlib
import os
import json
import logging
import re
import subprocess
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config

MAX_UTTERANCES_PER_CLASS = 2 ** 27 - 1

logger = logging.getLogger(__name__)

# ==========================================
# 1. Core data pipeline
# ==========================================

def load_raw_features():
    X = np.load(Config.FEATURE_FILE)
    y = np.load(Config.LABEL_FILE)
    label_names = np.load(Config.LABEL_MAP_FILE, allow_pickle=True)
    return X, y, label_names


def which_set(filename, val_pct, test_pct):
    """Assign a WAV file to train/val/test by speaker hash (Warden 2018, §7).

    All utterances of one speaker land in the same split, so a model cannot be
    scored on a speaker it was trained on.
    """
    base = os.path.basename(filename)
    speaker = re.sub(r"_nohash_.*$", "", base)
    digest = hashlib.sha1(speaker.encode("utf-8")).hexdigest()
    percentage = (int(digest, 16) % (MAX_UTTERANCES_PER_CLASS + 1)) * \
        (100.0 / MAX_UTTERANCES_PER_CLASS)
    if percentage < val_pct:
        return "val"
    if percentage < val_pct + test_pct:
        return "test"
    return "train"


def dataset_filenames(label_names):
    """Rebuild the per-row file list produced by collect_data.py.

    Row order is class-major, matching the label array; verified against the
    per-class sample counts before use.
    """
    from collect_data import SAMPLES_PER_CLASS

    paths = []
    for label in label_names:
        folder = os.path.join(Config.DATA_DIR, label)
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        paths.extend(os.path.join(folder, f) for f in files[:SAMPLES_PER_CLASS])
    return paths


def speaker_split_indices(y, label_names, val_size, test_size):
    # Extractors that record per-row paths give an exact mapping; otherwise the
    # file list is rebuilt and validated against the per-class counts.
    if Config.PATHS_FILE and os.path.exists(Config.PATHS_FILE):
        paths = list(np.load(Config.PATHS_FILE, allow_pickle=True))
    else:
        paths = dataset_filenames(label_names)
    if len(paths) != len(y):
        raise RuntimeError(
            f"Cannot reconstruct file list: {len(paths)} files vs {len(y)} feature rows. "
            "Re-run collect_data.py so features and dataset stay in sync."
        )
    for idx, label in enumerate(label_names):
        expected = int((y == idx).sum())
        found = sum(1 for p in paths if os.path.basename(os.path.dirname(p)) == label)
        if expected != found:
            raise RuntimeError(f"Class '{label}' count mismatch: {found} files vs {expected} rows.")

    assignment = np.array([which_set(p, val_size * 100, test_size * 100) for p in paths])
    return (np.where(assignment == "train")[0],
            np.where(assignment == "val")[0],
            np.where(assignment == "test")[0])


def split_and_scale(test_size=None, val_size=None, random_state=None,
                    use_delta_delta=False, speaker_split=False):
    X, y, label_names = load_raw_features()

    if test_size is None:
        test_size = Config.TEST_SPLIT
    if val_size is None:
        val_size = Config.VAL_SPLIT
    if random_state is None:
        random_state = Config.RANDOM_SEED

    # Add delta-delta features on the fly (before split, since it's per-sample)
    if use_delta_delta:
        X = compute_delta_delta(X)
        logger.info("Delta-delta features added: shape = " + str(X.shape))

    if speaker_split:
        tr_idx, val_idx, test_idx = speaker_split_indices(
            y, label_names, val_size, test_size)
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        logger.info("Speaker-disjoint split (Warden 2018 hashing)")
    else:
        # Step 1: Split off test set (unseen until final evaluation)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Step 2: Split remaining into train & val
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )

    # Step 3: Fit scaler on TRAINING SET ONLY and transform all splits
    num_samples, num_frames, num_features = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_features)
    scaler.fit(X_train_flat)

    X_train = scaler.transform(X_train_flat).reshape(num_samples, num_frames, num_features)
    X_val = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape[0], num_frames, num_features)
    X_test = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape[0], num_frames, num_features)

    # Step 4: Transpose to PyTorch format (Batch, Channels, Length)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    # Save scaler for inference reuse
    joblib.dump(scaler, Config.SCALER_FILE)
    logger.info(f"Scaler saved to {Config.SCALER_FILE}")

    num_features_out = X_train.shape[1]  # channels after transpose
    logger.info(
        f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, "
        f"features={num_features_out}, classes={len(label_names)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, label_names, scaler


def compute_delta_delta(X):
    """Add delta-delta (acceleration) features to existing (N, 40, 26) features.

    Assumes X has shape (N, frames, 26) = [13 MFCC, 13 delta].
    Returns X with shape (N, frames, 39) = [13 MFCC, 13 delta, 13 delta-delta].
    """
    half = X.shape[-1] // 2  # 13
    delta_part = X[:, :, half:]  # (N, frames, 13) - the delta features
    delta2 = np.diff(delta_part, axis=1, prepend=delta_part[:, :1, :])
    return np.concatenate([X, delta2], axis=-1)


def flatten_features(X):
    """Flatten (N, channels, frames) into (N, channels * frames) for dense models."""
    return X.reshape(X.shape[0], -1)


def load_scaler():
    return joblib.load(Config.SCALER_FILE)


def preprocess_single(file_path, scaler=None, use_delta_delta=False):
    """Extract MFCC from one WAV via analyzer.exe, apply scaling, return tensor ready for model."""
    result = subprocess.check_output(
        [Config.ANALYZER_PATH, file_path], stderr=subprocess.DEVNULL
    )
    data = json.loads(result)
    mfccs = np.array([f['mfcc'] for f in data])

    MAX_LEN = Config.NUM_FRAMES
    if len(mfccs) < MAX_LEN:
        mfccs = np.pad(mfccs, ((0, MAX_LEN - len(mfccs)), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:MAX_LEN, :]

    delta = np.diff(mfccs, axis=0, prepend=mfccs[:1])
    feat_matrix = np.hstack([mfccs, delta])

    if use_delta_delta:
        delta2 = np.diff(delta, axis=0, prepend=delta[:1])
        feat_matrix = np.hstack([feat_matrix, delta2])

    if scaler is not None:
        feat_matrix = scaler.transform(feat_matrix)

    feat_matrix = np.expand_dims(feat_matrix.T, axis=0)
    return feat_matrix.astype(np.float32)
