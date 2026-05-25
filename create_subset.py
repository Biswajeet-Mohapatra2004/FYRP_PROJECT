"""
Subset Creator for CASIA-WebFace
Creates a smaller data/clean_subset/ from the full data/clean/
by taking up to MAX_IMAGES_PER_IDENTITY images from
up to MAX_IDENTITIES identities.

Default: 500 identities × 10 images = ~5,000 images
→ Fast enough for CPU training demo / validation.
"""

import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────
SOURCE_DIR           = Path("data/clean")
SUBSET_DIR           = Path("data/clean_subset")
MAX_IDENTITIES       = 500      # number of identity folders to include
MAX_IMAGES_PER_IDENT = 10       # max images to copy per identity
MIN_IMAGES_PER_IDENT = 3        # skip identities with fewer images
SEED                 = 42

random.seed(SEED)

def create_subset():
    print("=" * 60)
    print(f"  CASIA-WebFace Subset Creator")
    print(f"  Source : {SOURCE_DIR.resolve()}")
    print(f"  Dest   : {SUBSET_DIR.resolve()}")
    print(f"  Config : {MAX_IDENTITIES} identities x {MAX_IMAGES_PER_IDENT} imgs")
    print("=" * 60)

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    # Gather all valid identity directories
    all_identities = sorted([
        d for d in SOURCE_DIR.iterdir()
        if d.is_dir() and len(list(d.glob("*.jpg"))) >= MIN_IMAGES_PER_IDENT
    ])
    print(f"\nFound {len(all_identities)} valid identities in source")

    # Sample a random subset
    selected = random.sample(all_identities, min(MAX_IDENTITIES, len(all_identities)))
    selected.sort()

    total_copied = 0
    for identity_dir in tqdm(selected, desc="Copying subset"):
        images = list(identity_dir.glob("*.jpg"))
        # Take up to MAX_IMAGES_PER_IDENT images
        chosen = random.sample(images, min(MAX_IMAGES_PER_IDENT, len(images)))

        dest = SUBSET_DIR / identity_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for img in chosen:
            shutil.copy2(img, dest / img.name)
            total_copied += 1

    print(f"\n{'=' * 60}")
    print(f"  Subset created in: {SUBSET_DIR.resolve()}")
    print(f"  Identities copied : {len(selected)}")
    print(f"  Total images      : {total_copied}")
    print(f"{'=' * 60}")
    print(f"\nTrain on subset:")
    print(f"  python train.py --mode clean --data_dir data/clean_subset")


if __name__ == "__main__":
    create_subset()
