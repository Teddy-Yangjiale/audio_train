import os
import json
import logging
import subprocess
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config

logger = logging.getLogger(__name__)

# ==========================================
# 1. Core data pipeline
# ==========================================

def load_raw_features():
    X = np.load(Config.FEATURE_FILE)
    y = np.load(Config.LABEL_FILE)
    label_names = np.load(Config.LABEL_MAP_FILE, allow_pickle=True)
    return X, y, label_names


def split_and_scale(test_size=None, val_size=None, random_state=None,
                    use_delta_delta=False):
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
