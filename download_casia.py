"""
CASIA-WebFace Dataset Downloader & Organizer
Uses kagglehub to download the dataset and organizes it into the
identity-subfolder structure expected by FaceDataset.

Usage:
    python download_casia.py

Dataset: debarghamitraroy/casia-webface
  ~500K images | ~10,575 identities | ~4 GB
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────
KAGGLE_DATASET   = "debarghamitraroy/casia-webface"
CLEAN_DIR        = Path("data/clean")
MIN_IMAGES       = 2        # keep identities with at least this many images

# ── Step 1: Download via kagglehub ──────────────────────────────────

def download():
    print("[1/3] Downloading CASIA-WebFace via kagglehub...")
    print(f"      Dataset : {KAGGLE_DATASET}")
    print("      Note    : Requires Kaggle credentials (kaggle.json)\n")

    try:
        import kagglehub
    except ImportError:
        print("  kagglehub not found — installing...")
        os.system("pip install kagglehub -q")
        import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"\n  Downloaded to: {path}")
    return Path(path)


# ── Step 2: Discover the root identity folder ───────────────────────

def find_identity_root(base: Path) -> Path:
    """
    Walk the downloaded directory to find the folder that directly
    contains identity sub-directories (folders whose children are images).
    """
    # Check base itself first
    subdirs = [d for d in base.iterdir() if d.is_dir()]

    if not subdirs:
        raise RuntimeError(f"No sub-directories found under {base}")

    # If the first subdir contains images → base IS the identity root
    sample = subdirs[0]
    images_in_sample = list(sample.glob("*.jpg")) + list(sample.glob("*.png"))
    if images_in_sample:
        return base

    # Otherwise go one level deeper (common when zip has a wrapper folder)
    for d in subdirs:
        inner = [x for x in d.iterdir() if x.is_dir()]
        if inner:
            sample2 = inner[0]
            imgs2 = list(sample2.glob("*.jpg")) + list(sample2.glob("*.png"))
            if imgs2:
                return d

    # Fallback: just return base and let the copy handle it
    return base


# ── Step 3: Organize into data/clean/ ──────────────────────────────

def organize(source_root: Path):
    print(f"\n[2/3] Organizing dataset → {CLEAN_DIR.resolve()}")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    identity_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()])
    print(f"      Found {len(identity_dirs)} identities in source\n")

    kept = skipped = total_images = 0

    for identity_dir in tqdm(identity_dirs, desc="Copying identities"):
        images = (
            list(identity_dir.glob("*.jpg")) +
            list(identity_dir.glob("*.jpeg")) +
            list(identity_dir.glob("*.png"))
        )

        if len(images) < MIN_IMAGES:
            skipped += 1
            continue

        dest = CLEAN_DIR / identity_dir.name
        dest.mkdir(parents=True, exist_ok=True)

        for img in images:
            target = dest / img.name
            if not target.exists():
                shutil.copy2(img, target)

        kept += 1
        total_images += len(images)

    return kept, skipped, total_images


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CASIA-WebFace Dataset Downloader & Organizer")
    print("=" * 60)

    # 1. Download
    download_path = download()

    # 2. Find the right root
    print("\n[2/3] Locating identity folders...")
    identity_root = find_identity_root(download_path)
    print(f"      Identity root: {identity_root}")

    # 3. Organize
    kept, skipped, total = organize(identity_root)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  DONE — CASIA-WebFace ready in: {CLEAN_DIR.resolve()}")
    print(f"  Identities kept    : {kept}  (>= {MIN_IMAGES} images)")
    print(f"  Identities skipped : {skipped}  (< {MIN_IMAGES} images)")
    print(f"  Total images       : {total:,}")
    print(f"{'=' * 60}")
    print(f"\nNext step — start Phase 1 training:")
    print(f"  python train.py --mode clean --data_dir data/clean")
    print(f"  (add --epochs 30 --batch_size 64 for full training)")
