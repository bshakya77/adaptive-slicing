#!/usr/bin/env python3
from pathlib import Path

# ← Update these paths to your two directories
folder1 = Path("runs/slice_coco/subset_train_sliced_images_3000/images")
folder2 = Path("runs/slice_coco/subset_train_sliced_images_3000/labels")

def sync_folders(dir_a: Path, dir_b: Path):
    # 1. Collect all file‐name stems in each directory
    stems_a = {p.stem for p in dir_a.iterdir() if p.is_file()}
    stems_b = {p.stem for p in dir_b.iterdir() if p.is_file()}

    # 2. Compute the common stems
    common = stems_a & stems_b

    # 3. Remove any file in dir_a not in common
    for f in dir_a.iterdir():
        if f.is_file() and f.stem not in common:
            f.unlink()
            print(f"Removed {f.name} from {dir_a}")

    # 4. Remove any file in dir_b not in common
    for f in dir_b.iterdir():
        if f.is_file() and f.stem not in common:
            f.unlink()
            print(f"Removed {f.name} from {dir_b}")

if __name__ == "__main__":
    # sanity checks
    for d in (folder1, folder2):
        if not d.is_dir():
            raise SystemExit(f"ERROR: not a directory: {d}")

    sync_folders(folder1, folder2)
    print("Sync complete — only matched files remain.")
