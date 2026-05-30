"""
Configuration for the Adversarial Face Detection System.
All hyperparameters, paths, and settings in one place.
"""

import os


class Config:
    """Central configuration for the entire CNN pipeline."""

    # ──────────────────────────── Paths ────────────────────────────
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CLEAN_DIR = os.path.join(DATA_DIR, "clean")
    ADVERSARIAL_DIR = os.path.join(DATA_DIR, "adversarial")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    # ──────────────────────────── Model ────────────────────────────
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 3          # normal, adversarial, suspicious
    CLASS_NAMES = ["normal", "adversarial", "suspicious"]
    EMBEDDING_DIM = 512      # ResNet-18 final feature dimension
    PRETRAINED = True        # Use ImageNet pretrained weights

    # ──────────────────────────── Image ────────────────────────────
    IMAGE_SIZE = 224         # ResNet standard input size
    IMAGE_CHANNELS = 3
    # ImageNet normalization (used because backbone is pretrained on ImageNet)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # ──────────────────────────── Training ─────────────────────────
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    SCHEDULER_STEP = 10
    SCHEDULER_GAMMA = 0.1
    EARLY_STOPPING_PATIENCE = 7
    VALIDATION_SPLIT = 0.2
    NUM_WORKERS = 0          # 0 = safe on Windows (avoids DataLoader multiprocessing hangs)

    # ──────────────────────────── Adversarial Attacks ──────────────
    FGSM_EPSILON = 0.03      # Perturbation budget for FGSM
    PGD_EPSILON = 0.03       # Perturbation budget for PGD
    PGD_ALPHA = 0.003        # Step size for PGD (≈ epsilon/10 at default epsilon=0.03)
    PGD_STEPS = 40           # Number of PGD iterations (more steps = stronger attack)

    # ──────────────────────────── Thresholds ───────────────────────
    CONFIDENCE_THRESHOLD = 0.5
    SUSPICIOUS_LOW = 0.3     # Below this → suspicious
    SUSPICIOUS_HIGH = 0.7    # Between low and high → suspicious

    # ──────────────────────────── Device ───────────────────────────
    DEVICE = "cuda"          # Will fallback to cpu in code if not available

    @classmethod
    def create_dirs(cls):
        """Create all necessary directories if they don't exist."""
        for d in [cls.DATA_DIR, cls.CLEAN_DIR, cls.ADVERSARIAL_DIR,
                  cls.MODEL_SAVE_DIR, cls.LOGS_DIR, cls.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)
