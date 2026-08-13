# Audio Bandwidth Extension config
import os
import random
import torch
import numpy as np


class BWEConfig:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    # Sample rates
    LOW_SR = 8000
    HIGH_SR = 16000
    SEGMENT_SEC = 0.5               # seconds per training segment
    SEGMENT_SAMPLES = int(HIGH_SR * SEGMENT_SEC)  # 8000

    # Model
    BASE_CHANNELS = 64
    NUM_ENCODERS = 5
    KERNEL_SIZE = 15
    STRIDE = 4                       # down/up-sample stride

    # Training
    BATCH_SIZE = 8
    EPOCHS = 200
    LR = 0.0002
    WEIGHT_DECAY = 0.0
    GRAD_CLIP = 1.0
    EARLY_STOP_PATIENCE = 30
    LR_PATIENCE = 4                  # epochs without val improvement before LR is halved

    # Loss weights
    L1_WEIGHT = 100.0
    STFT_WEIGHT = 1.0
    STFT_SCALES = [2048, 1024, 512]   # multi-resolution FFT sizes
    STFT_HOP_RATIOS = [0.25, 0.25, 0.25]  # hop = fft * ratio

    # Data
    DATA_DIR = os.path.join(PROJECT_ROOT, "speech_commands")
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42

    # Checkpoint
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_bwe")
    LOG_FILE = os.path.join(PROJECT_ROOT, "train_bwe.log")

    @classmethod
    def set_seed(cls, seed=None):
        if seed is None:
            seed = cls.RANDOM_SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @classmethod
    def init_env(cls):
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')
