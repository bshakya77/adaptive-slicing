#!/usr/bin/env python3
"""
Simple script to copy COCO annotation .json files whose names match images.

Edit the three path variables below to point to your folders.
"""

from pathlib import Path
import shutil

def copy_matched_annotations(annotations_dir: Path, images_dir: Path, output_dir: Path):
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all image base names (without extension)
    image_stems = {p.stem for p in images_dir.rglob('*') if p.is_file()}

    # Copy matching annotation files
    for ann_file in annotations_dir.glob('*.txt'):
        if ann_file.stem in image_stems:
            dest = output_dir / ann_file.name
            shutil.copy2(ann_file, dest)
            print(f"Copied: {ann_file.name}")

    # Notify if none were copied
    if not any(output_dir.iterdir()):
        print("No matching annotation files found.")
    
    print("Completed !!")
    
if __name__ == "__main__":
    # <-- Update these paths -->
    annotations_dir = Path("../yolov9/datasets/VisDrone/VisDrone2019-DET-test-dev/labels")
    images_dir      = Path("./custom_utils/testdev_subsets/mix/v1/images")
    output_dir      = Path("./custom_utils/testdev_subsets/mix/v1/labels")
    # --------------------------

    copy_matched_annotations(annotations_dir, images_dir, output_dir)
