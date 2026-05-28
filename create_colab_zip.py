"""
Creates a clean FYRP_code.zip for Colab upload.
Only includes source code — excludes venv, checkpoints, data, results, logs, cache.

Run from the project root:
    python create_colab_zip.py
"""

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent

# Folders to skip entirely
SKIP_DIRS = {
    'venv', '.venv', 'env',
    'checkpoints',
    'data',
    'results',
    'logs',
    '__pycache__',
    '.git', '.github',
    'node_modules',
}

# File extensions to skip
SKIP_EXTS = {
    '.pyc', '.pyo',          # compiled Python
    '.pth', '.pt', '.ckpt',  # model weights
    '.zip', '.tar', '.gz',   # archives
    '.jpg', '.jpeg', '.png', '.gif', '.bmp',  # images in root
    '.pdf',                  # presentation files
    '.log',                  # log files
    '.DS_Store',
}

# Specific filenames to skip
SKIP_FILES = {
    '.env',           # secrets — use .env.example instead
    'FYRP_code.zip',
}

def should_skip(path: Path) -> bool:
    # Skip excluded directories anywhere in the path
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    if path.name in SKIP_FILES:
        return True
    if path.suffix.lower() in SKIP_EXTS:
        # Allow .py files regardless (extension check below handles this)
        return True
    return False

def collect_files():
    files = []
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        parts = rel.parts

        # Skip if any parent dir is in SKIP_DIRS
        if any(p in SKIP_DIRS for p in parts[:-1]):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix.lower() in SKIP_EXTS:
            continue

        files.append((path, rel))
    return files

def main():
    out_path = ROOT / 'FYRP_code.zip'
    files = collect_files()

    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for abs_path, rel_path in sorted(files):
            zf.write(abs_path, rel_path)
            print(f"  + {rel_path}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nCreated: {out_path}")
    print(f"Files included: {len(files)}")
    print(f"Size: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
