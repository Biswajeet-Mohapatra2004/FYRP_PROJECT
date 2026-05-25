"""
Dataset Downloader
Downloads LFW (Labeled Faces in the Wild) dataset using torchvision
and organises it into the identity-folder structure expected by FaceDataset.

LFW details:
  - 13,233 images of 5,749 identities
  - Freely available, no registration required
  - ~173 MB download
  - Source: http://vis-www.cs.umass.edu/lfw/

Folder structure produced:
  data/
    clean/
      person_name_1/
        img_01.jpg
        img_02.jpg
      person_name_2/
        ...
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────
LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
DOWNLOAD_DIR = Path("data/_download")
CLEAN_DIR    = Path("data/clean")
MIN_IMAGES_PER_IDENTITY = 2   # keep only identities with >= 2 images


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_lfw():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tgz_path = DOWNLOAD_DIR / "lfw.tgz"

    # ── Step 1: Download ──────────────────────────────────────────
    if tgz_path.exists():
        print(f"[OK] Archive already exists: {tgz_path}")
    else:
        print(f"[...] Downloading LFW dataset (~173 MB)...")
        print(f"   Source: {LFW_URL}\n")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1,
                                 desc="lfw.tgz") as t:
            urllib.request.urlretrieve(LFW_URL, tgz_path, reporthook=t.update_to)
        print(f"\n   Saved → {tgz_path}")

    # ── Step 2: Extract ───────────────────────────────────────────
    extract_dir = DOWNLOAD_DIR / "lfw"
    if extract_dir.exists():
        print(f"[OK] Already extracted: {extract_dir}")
    else:
        print(f"\n[...] Extracting archive...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(DOWNLOAD_DIR)
        print(f"   Extracted → {extract_dir}")

    # ── Step 3: Organise into data/clean/ ────────────────────────
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    identity_dirs = sorted([d for d in extract_dir.iterdir() if d.is_dir()])
    print(f"\n[INFO] Found {len(identity_dirs)} identities in LFW")

    kept = 0
    skipped = 0
    total_images = 0

    for identity_dir in tqdm(identity_dirs, desc="Organising"):
        images = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))

        if len(images) < MIN_IMAGES_PER_IDENTITY:
            skipped += 1
            continue

        dest_dir = CLEAN_DIR / identity_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            dest = dest_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)

        kept += 1
        total_images += len(images)

    print(f"\n{'='*55}")
    print(f"[DONE] LFW dataset ready in: {CLEAN_DIR.resolve()}")
    print(f"   Identities kept : {kept}  (>= {MIN_IMAGES_PER_IDENTITY} images)")
    print(f"   Identities skipped (1 image only): {skipped}")
    print(f"   Total images    : {total_images}")
    print(f"{'='*55}")
    print(f"\nNext step:")
    print(f"   Activate venv:  venv\\Scripts\\activate")
    print(f"   Phase 1 train:  python train.py --mode clean --data_dir data/clean")


if __name__ == "__main__":
    download_lfw()
