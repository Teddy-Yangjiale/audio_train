import os
import random
import torch
import numpy as np

class Config:
    DATA_DIR = "./speech_commands"
    ANALYZER_PATH = "./analyzer.exe"
    FEATURE_FILE = "X_features.npy"
    LABEL_FILE = "y_labels.npy"
    LABEL_MAP_FILE = "label_map.npy"
    PATHS_FILE = None            # per-row WAV paths, when the extractor saved them
    SCALER_FILE = "scaler.pkl"
    EXCLUDE = ["_background_noise_"]

    NUM_FRAMES = 40
    NUM_MFCC = 13

    # Feature sets: "analyzer" = analyzer.exe (125 ms hop, ~8 frames/clip),
    #               "torch"    = torchaudio (10 ms hop, 101 frames/clip)
    # (features, labels, label map, paths, frames, time-shift frames ≈ ±100 ms)
    FEATURE_SETS = {
        "analyzer": ("X_features.npy", "y_labels.npy", "label_map.npy", None, 40, 1),
        "torch": ("X_features_torch.npy", "y_labels_torch.npy",
                  "label_map_torch.npy", "paths_torch.npy", 101, 10),
    }
    TIME_SHIFT_FRAMES = 1

    # Training
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 0.001
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.0       # set to 0.1 for label smoothing
    USE_CLASS_WEIGHTS = False    # compute class weights for imbalanced data

    # Early stopping
    EARLY_STOP_PATIENCE = 15
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    GRAD_CLIP = 1.0

    # Scheduler: "plateau" or "cosine"
    SCHEDULER = "plateau"
    COSINE_T_MAX = 100

    # Data
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42

    # Checkpoint
    CHECKPOINT_DIR = "checkpoints"
    LOG_FILE = "train.log"
    SAVE_BEST_ONLY = True       # if False, save every epoch

    @classmethod
    def use_feature_set(cls, name):
        if name not in cls.FEATURE_SETS:
            raise ValueError(f"Unknown feature set '{name}'. "
                             f"Available: {list(cls.FEATURE_SETS)}")
        (cls.FEATURE_FILE, cls.LABEL_FILE, cls.LABEL_MAP_FILE,
         cls.PATHS_FILE, cls.NUM_FRAMES, cls.TIME_SHIFT_FRAMES) = cls.FEATURE_SETS[name]
        cls.SCALER_FILE = f"scaler_{name}.pkl" if name != "analyzer" else "scaler.pkl"

    @classmethod
    def set_seed(cls, seed=None):
        if seed is None:
            seed = cls.RANDOM_SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def init_env(cls):
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')
        try:
            import torch.fx.experimental.sym_node
            if not hasattr(torch.fx.experimental.sym_node, 'DynamicInt'):
                torch.fx.experimental.sym_node.DynamicInt = int
        except Exception:
            pass
