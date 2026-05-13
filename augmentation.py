import numpy as np


def time_mask(feat_matrix, max_mask_frames=10, num_masks=1):
    """Apply time masking to 2D feature matrix (frames, features)."""
    matrix = feat_matrix.copy()
    num_frames = matrix.shape[0]
    for _ in range(num_masks):
        t = np.random.randint(0, max_mask_frames)
        t0 = np.random.randint(0, num_frames - t) if num_frames > t else 0
        matrix[t0:t0 + t, :] = 0
    return matrix


def freq_mask(feat_matrix, max_mask_bins=8, num_masks=1):
    """Apply frequency masking to 2D feature matrix (frames, features)."""
    matrix = feat_matrix.copy()
    num_bins = matrix.shape[1]
    for _ in range(num_masks):
        f = np.random.randint(0, max_mask_bins)
        f0 = np.random.randint(0, num_bins - f) if num_bins > f else 0
        matrix[:, f0:f0 + f] = 0
    return matrix


def time_shift(feat_matrix, max_shift_frames=5):
    """Shift the feature matrix in the time dimension with zero-padding."""
    matrix = feat_matrix.copy()
    shift = np.random.randint(-max_shift_frames, max_shift_frames + 1)
    if shift == 0:
        return matrix
    shifted = np.zeros_like(matrix, dtype=matrix.dtype)
    if shift > 0:
        shifted[shift:, :] = matrix[:-shift, :]
    else:
        shift = abs(shift)
        shifted[:-shift, :] = matrix[shift:, :]
    return shifted


def spec_augment(feat_matrix, freq_masks=1, time_masks=1):
    """SpecAugment: apply time + frequency masking."""
    matrix = time_mask(feat_matrix, num_masks=time_masks)
    matrix = freq_mask(matrix, num_masks=freq_masks)
    return matrix


_AUGMENTS = ["time_mask", "freq_mask", "time_shift", "spec_augment"]


def apply_augment(feat_matrix, augment_names=None, prob=0.5):
    """Apply random augmentation(s) with given probability.

    Args:
        feat_matrix: ndarray of shape (frames, features)
        augment_names: list of augment names, or None for a random one
        prob: probability of applying any augmentation
    """
    if np.random.random() > prob:
        return feat_matrix
    if augment_names is None:
        augment_names = [np.random.choice(_AUGMENTS)]
    result = feat_matrix.copy()
    for name in augment_names:
        if name == "time_mask":
            result = time_mask(result)
        elif name == "freq_mask":
            result = freq_mask(result)
        elif name == "time_shift":
            result = time_shift(result)
        elif name == "spec_augment":
            result = spec_augment(result)
    return result
