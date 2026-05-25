"""
MXNet RecordIO Extractor for CASIA-WebFace dataset.
Reads train.idx (offsets), train.lst (identity info), train.rec (image bytes)
and writes images to data/clean/<identity_id>/<seq>.jpg

Usage:
    python extract_casia.py

Expected output:
    data/clean/
        0000045/
            000001.jpg
            000002.jpg
        0000099/
            ...
"""

import os
import struct
from pathlib import Path
from tqdm import tqdm

# ── Subset controls ─────────────────────────────────────────────────
# Set to None to extract everything (full 490K images, ~15-20 min)
# For quick training / resource saving, use small limits:
MAX_IDENTITIES    = 2000   # max number of identity folders to extract
MAX_PER_IDENTITY  = 5      # max images per identity  (2000 × 5 = ~10K images)

# ── Auto-detect kagglehub cache path ────────────────────────────────
def _find_rec_dir() -> Path:
    """
    Try several known kagglehub cache locations to find train.rec.
    Works regardless of which Windows username was used to download.
    """
    base = Path.home() / ".cache" / "kagglehub" / "datasets" \
           / "debarghamitraroy" / "casia-webface" / "versions" / "2"

    candidates = [
        base / "casia-webface",   # sometimes extracted into a sub-folder
        base,                     # sometimes files sit directly here
    ]
    for c in candidates:
        if (c / "train.rec").exists():
            return c

    raise FileNotFoundError(
        f"Could not find train.rec under {base}.\n"
        "Run python download_casia.py first, or check your kagglehub cache."
    )

REC_DIR  = _find_rec_dir()
REC_FILE = REC_DIR / "train.rec"
IDX_FILE = REC_DIR / "train.idx"
LST_FILE = REC_DIR / "train.lst"
CLEAN_DIR = Path("data/clean")

# MXNet RecordIO magic constant
KMAGIC = 0xced7230a

# ── Step 1: Load offsets from .idx ──────────────────────────────────

def load_idx(idx_path: Path) -> dict:
    """Returns {index: byte_offset} from train.idx (text format: index\toffset)."""
    offsets = {}
    with open(idx_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                offsets[int(parts[0])] = int(parts[1])
    return offsets


# ── Step 2: Load identity mapping from .lst ─────────────────────────

def load_lst(lst_path: Path) -> dict:
    """
    Returns {index: identity_name} from train.lst.
    LST format: label\\tpath\\t...
    Path example: /raid5data/dplearn/CASIA-WebFace/0000045/001.jpg
    Identity = the folder name (0000045)
    """
    identity_map = {}
    with open(lst_path, "r") as f:
        for idx_line, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            path_str = parts[1]                      # e.g. .../0000045/001.jpg
            identity = Path(path_str).parent.name    # e.g. 0000045
            # IDX file keys start at 1; lst is 0-indexed
            identity_map[idx_line + 1] = identity
    return identity_map


# ── Step 3: Read a single RecordIO record ───────────────────────────

def read_record(rec_file, offset: int):
    """
    Reads one MXNet RecordIO record from rec_file at the given byte offset.

    RecordIO record layout (from inspection of actual data):
        4 bytes : magic (0xced7230a, LE)
        4 bytes : cflag  (upper 29 bits = payload length, lower 3 bits = flag)
        24 bytes: extended header (flag * float32 labels + 2x uint64 ids)
        rest    : JPEG image bytes

    From probing: skip=24 bytes into payload reveals the JPEG header (0xFFD8).
    """
    rec_file.seek(offset)
    header = rec_file.read(8)
    if len(header) < 8:
        return None

    magic, cflag = struct.unpack("<II", header)
    if magic != KMAGIC:
        return None

    length = cflag >> 3          # upper 29 bits = payload length
    if length <= 24:
        return None

    payload = rec_file.read(length)
    if len(payload) < 26:
        return None

    # Skip the 24-byte extended header (confirmed by byte inspection)
    image_bytes = payload[24:]

    # Validate JPEG signature
    if image_bytes[:2] != b"\xff\xd8":
        return None

    return image_bytes



# ── Step 4: Extract all images ──────────────────────────────────────

def extract():
    print("=" * 60)
    print("  CASIA-WebFace RecordIO Extractor")
    print("=" * 60)

    print("\n[1/4] Loading offset index...")
    offsets = load_idx(IDX_FILE)
    print(f"      {len(offsets):,} records indexed")

    print("[2/4] Loading identity map from .lst...")
    identity_map = load_lst(LST_FILE)
    print(f"      {len(identity_map):,} entries loaded")

    print(f"[3/4] Extracting images to {CLEAN_DIR.resolve()}")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    errors  = 0
    img_counter: dict = {}   # identity -> count
    identity_seen = set()    # identities we've started writing

    sorted_keys = sorted(offsets.keys())

    if MAX_IDENTITIES is not None:
        print(f"      Subset mode: max {MAX_IDENTITIES} identities, "
              f"{MAX_PER_IDENTITY} imgs each (~{MAX_IDENTITIES*MAX_PER_IDENTITY:,} images)")

    with open(REC_FILE, "rb") as rec_f:
        for key in tqdm(sorted_keys, desc="Extracting"):
            identity = identity_map.get(key)
            if identity is None:
                skipped += 1
                continue

            # ── Subset gate: max identities ──────────────────────────
            if MAX_IDENTITIES is not None:
                if identity not in identity_seen and len(identity_seen) >= MAX_IDENTITIES:
                    skipped += 1
                    continue

            # ── Subset gate: max images per identity ─────────────────
            current_count = img_counter.get(identity, 0)
            if MAX_PER_IDENTITY is not None and current_count >= MAX_PER_IDENTITY:
                skipped += 1
                continue

            identity_seen.add(identity)

            offset = offsets[key]
            image_bytes = read_record(rec_f, offset)


            if image_bytes is None:
                errors += 1
                continue

            # Build output path
            dest_dir = CLEAN_DIR / identity
            dest_dir.mkdir(parents=True, exist_ok=True)

            img_counter[identity] = img_counter.get(identity, 0) + 1
            fname = f"{img_counter[identity]:06d}.jpg"
            dest_path = dest_dir / fname

            if not dest_path.exists():
                with open(dest_path, "wb") as f:
                    f.write(image_bytes)
            success += 1

    identities_found = len(img_counter)
    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Images extracted : {success:,}")
    print(f"  Identities found : {identities_found:,}")
    print(f"  Skipped (no map) : {skipped:,}")
    print(f"  Errors           : {errors:,}")
    print(f"  Output dir       : {CLEAN_DIR.resolve()}")
    print(f"{'=' * 60}")
    print(f"\nNext step:")
    print(f"  python train.py --mode clean --data_dir data/clean")
    return success, identities_found


if __name__ == "__main__":
    extract()
