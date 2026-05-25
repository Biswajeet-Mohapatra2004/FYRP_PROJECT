"""
Main training script.

Run (Phase 1 — train on clean data first):
    python train.py --mode clean --data_dir data/clean

Run (Phase 2 — fine-tune on mixed adversarial dataset):
    python train.py --mode adversarial --csv data/labels.csv

Run (Generate adversarial dataset from a trained clean model):
    python train.py --mode generate --data_dir data/clean \
                    --checkpoint checkpoints/best_model.pth
"""

import argparse
import logging
import os
import sys

import torch

from model.config import Config
from model.cnn_model import AdversarialFaceDetector, model_summary
from model.dataset import DatasetBuilder
from model.trainer import Trainer
from model.adversarial import AdversarialDatasetGenerator


os.makedirs("logs", exist_ok=True)       # ensure logs/ exists before FileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Argument parser
# ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / fine-tune the Adversarial Face Detector (ResNet-18)"
    )
    parser.add_argument(
        "--mode",
        choices=["clean", "adversarial", "generate"],
        default="clean",
        help=(
            "clean      → train on clean images only (Phase 1)\n"
            "adversarial→ fine-tune on mixed clean+adversarial (Phase 2)\n"
            "generate   → generate adversarial dataset from trained model"
        ),
    )
    parser.add_argument("--data_dir", type=str, default=Config.CLEAN_DIR,
                        help="Path to clean image directory (identity sub-dirs)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to labels.csv for adversarial mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint to load (for generate / fine-tune)")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (auto-detected if not set)")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────
#  Phase 1: train on clean images (binary: normal vs. adversarial not yet learnt)
# ────────────────────────────────────────────────────────────────────

def run_clean_training(args):
    logger.info("=== PHASE 1: Clean-data training ===")
    Config.create_dirs()
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr

    builder = DatasetBuilder.from_directory(args.data_dir)
    logger.info(f"Dataset: {builder}")

    model_summary()
    trainer = Trainer(device=args.device, learning_rate=args.lr)

    history = trainer.fit(builder.train_loader, builder.val_loader, epochs=args.epochs)
    results = trainer.evaluate(builder.test_loader)

    logger.info("Phase 1 complete.")
    return trainer.model, results


# ────────────────────────────────────────────────────────────────────
#  Phase 2: fine-tune on mixed adversarial dataset
# ────────────────────────────────────────────────────────────────────

def run_adversarial_training(args):
    logger.info("=== PHASE 2: Adversarial fine-tuning ===")

    if not args.csv:
        raise ValueError("--csv is required for adversarial mode. "
                         "Run --mode generate first.")

    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr

    # Load checkpoint if given
    model = AdversarialFaceDetector()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        sd = state.get("model_state_dict", state)
        model.load_state_dict(sd)
        logger.info(f"Loaded weights from {args.checkpoint}")

    builder = DatasetBuilder.from_csv(args.csv)
    logger.info(f"Dataset: {builder}")

    model_summary(model)
    trainer = Trainer(model=model, device=args.device, learning_rate=args.lr)

    history = trainer.fit(builder.train_loader, builder.val_loader, epochs=args.epochs)
    results = trainer.evaluate(builder.test_loader)

    logger.info("Phase 2 complete.")
    return trainer.model, results


# ────────────────────────────────────────────────────────────────────
#  Dataset generation
# ────────────────────────────────────────────────────────────────────

def run_generate(args):
    logger.info("=== Generating adversarial dataset ===")

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for generate mode.")

    # Load trained model
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AdversarialFaceDetector()
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    logger.info(f"Loaded model from {args.checkpoint}")

    # Build clean loader (no augmentation — we want clean images as base)
    from model.dataset import FaceDataset, get_eval_transform
    from torch.utils.data import DataLoader
    ds = FaceDataset(args.data_dir, transform=get_eval_transform(), label_type="binary")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    generator = AdversarialDatasetGenerator(
        model=model,
        clean_loader=loader,
        output_dir=Config.DATA_DIR,
        attack_types=["fgsm", "pgd"],
        device=device,
    )
    csv_path = generator.generate(denormalize=True)
    logger.info(f"CSV written to: {csv_path}")
    print(f"\n[Done] Next step: run  python train.py --mode adversarial --csv {csv_path}")


# ────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    dispatch = {
        "clean": run_clean_training,
        "adversarial": run_adversarial_training,
        "generate": run_generate,
    }

    dispatch[args.mode](args)
