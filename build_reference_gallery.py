"""
Build Reference Gallery — one-time utility script

Loads the trained checkpoint and extracts 512-d embeddings from clean training
images. Saves them to checkpoints/reference_gallery.pt for use by the
DynamicThresholdEngine drift calculation in pipeline.py.

Usage:
    # From full clean dataset (data/clean/):
    python build_reference_gallery.py

    # From a specific folder or the project root (for quick testing):
    python build_reference_gallery.py --clean_dir .

    # From individual image files:
    python build_reference_gallery.py --images tony_stark.jpg images.jpg

Output:
    checkpoints/reference_gallery.pt
"""

import os
import pickle
import argparse
import logging
import random

import torch
from pathlib import Path
from tqdm import tqdm

from model.config import Config
from model.cnn_model import FeatureExtractor
from utils.preprocessing import preprocess_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_gallery(
    checkpoint: str,
    clean_dir: str,
    output_path: str,
    n_samples: int = 1000,
    seed: int = 42,
    extra_images: list = None,
):
    """
    Extract embeddings from clean images and save to disk.

    Args:
        checkpoint    : path to trained model .pth file
        clean_dir     : root directory of clean images (searched recursively)
        output_path   : where to save the gallery (.pt file)
        n_samples     : how many clean images to sample
        seed          : random seed for reproducible sampling
        extra_images  : additional individual image paths to always include
    """
    random.seed(seed)

    # Collect all image paths recursively, excluding adversarial/results/venv dirs
    clean_path = Path(clean_dir)
    all_images = []
    skip_dirs = {"venv", "results", "adversarial", "__pycache__", ".git"}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in clean_path.rglob(ext):
            if not any(part in skip_dirs for part in p.parts):
                all_images.append(p)

    # Add individually specified images
    if extra_images:
        for img in extra_images:
            p = Path(img)
            if p.exists():
                all_images.append(p)
                logger.info(f"Added individual image: {p}")
            else:
                logger.warning(f"Image not found, skipping: {img}")

    if not all_images:
        logger.error(
            f"No images found in '{clean_dir}'.\n"
            "  • If your dataset isn't downloaded yet, try:\n"
            "      python build_reference_gallery.py --clean_dir . --n 10\n"
            "  • Or specify individual clean images:\n"
            "      python build_reference_gallery.py --images tony_stark.jpg images.jpg"
        )
        return

    logger.info(f"Found {len(all_images)} clean images total")

    # Sample up to n_samples
    sampled = random.sample(all_images, min(n_samples, len(all_images)))
    logger.info(f"Sampling {len(sampled)} images for the reference gallery")

    extractor = FeatureExtractor(checkpoint_path=checkpoint, device=None)
    extractor.model.eval()

    embeddings = []
    failed = 0

    for img_path in tqdm(sampled, desc="Extracting embeddings"):
        try:
            tensor = preprocess_image(str(img_path))
            with torch.no_grad():
                feature_dict = extractor.extract(tensor)
            embeddings.append(feature_dict["embedding"])
        except Exception as e:
            logger.warning(f"Skipping {img_path.name}: {e}")
            failed += 1

    logger.info(f"Extracted {len(embeddings)} embeddings ({failed} failed)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    logger.info(f"Reference gallery saved to: {output_path}")
    logger.info(f"Gallery size: {len(embeddings)} x 512-d embeddings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reference embedding gallery")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--clean_dir",
        default=Config.CLEAN_DIR,
        help="Root directory of clean images",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(Config.MODEL_SAVE_DIR, "reference_gallery.pt"),
        help="Output path for the gallery file",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of images to sample (default: 1000)",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Individual clean image files to include (e.g. tony_stark.jpg images.jpg)",
    )
    args = parser.parse_args()

    build_gallery(
        checkpoint=args.checkpoint,
        clean_dir=args.clean_dir,
        output_path=args.output,
        n_samples=args.n,
        extra_images=args.images,
    )
