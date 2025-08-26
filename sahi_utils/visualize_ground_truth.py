#!/usr/bin/env python3
"""
Ground Truth Visualization Tool
Refactored from predict.py to visualize only ground truth bounding boxes
"""

import logging
import os
import time
from typing import List, Optional, Dict, Union
from pathlib import Path

from PIL import Image
import numpy as np
from tqdm import tqdm

from sahi.utils.coco import Coco, CocoImage
from sahi.utils.cv import (
    IMAGE_EXTENSIONS,
    read_image_as_pil,
    visualize_object_predictions,
)
from sahi.utils.file import increment_path, list_files, save_json
from sahi.prediction import ObjectPrediction

logger = logging.getLogger(__name__)

def visualize_ground_truth_only(
    dataset_json_path: str,
    source: str = None,
    project: str = "runs/ground_truth",
    name: str = "exp",
    visual_bbox_thickness: int = 2,
    visual_text_size: float = 1.0,
    visual_text_thickness: int = 2,
    visual_hide_labels: bool = False,
    visual_export_format: str = "png",
    verbose: int = 1,
    return_dict: bool = False,
    **kwargs,
):
    """
    Visualizes only ground truth bounding boxes from provided COCO annotations.
    
    Args:
        dataset_json_path: str
            Path to COCO format annotation file (.json)
        source: str
            Folder directory that contains images or path of the image to be visualized.
            If None, will use image paths from the COCO annotations.
        project: str
            Save results to project/name.
        name: str
            Save results to project/name.
        visual_bbox_thickness: int
            Thickness of bounding box lines.
        visual_text_size: float
            Size of text labels.
        visual_text_thickness: int
            Thickness of text labels.
        visual_hide_labels: bool
            Hide class labels on bounding boxes.
        visual_export_format: str
            Export format for images ('jpg' or 'png').
        verbose: int
            0: no print
            1: print processing information
        return_dict: bool
            If True, returns a dict with 'export_dir' field.
    """
    
    # for profiling
    durations_in_seconds = dict()
    
    # init export directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
    visual_dir = save_dir / "ground_truth_visuals"
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    
    # load COCO annotations
    time_start = time.time()
    coco: Coco = Coco.from_coco_dict_or_path(dataset_json_path)
    time_end = time.time() - time_start
    durations_in_seconds["coco_load"] = time_end
    
    if verbose:
        # Count total annotations across all images
        total_annotations = sum(len(coco_image.annotations) for coco_image in coco.images)
        print(f"Loaded COCO annotations with {len(coco.images)} images and {total_annotations} annotations")
    
    # determine image paths
    if source:
        # Use provided source directory
        if os.path.isdir(source):
            image_iterator = list_files(
                directory=source,
                contains=IMAGE_EXTENSIONS,
                verbose=verbose,
            )
            # Create mapping from filename to full path
            image_path_mapping = {Path(img_path).name: img_path for img_path in image_iterator}
        else:
            # Single image file
            image_path_mapping = {Path(source).name: source}
    else:
        # Use image paths from COCO annotations
        image_path_mapping = {}
        for coco_image in coco.images:
            image_path_mapping[coco_image.file_name] = coco_image.file_name
    
    # iterate over COCO images
    durations_in_seconds["visualization"] = 0
    processed_count = 0
    
    for ind, coco_image in enumerate(
        tqdm(coco.images, f"Visualizing ground truth annotations", total=len(coco.images))
    ):
        # get filename - extract only the filename with extension, removing any directory path
        filename = Path(coco_image.file_name).name  # This gets only the filename with extension
        filename_without_extension = Path(filename).stem

        print("Filename:", filename)
        
        # find image path
        if filename in image_path_mapping:
            image_path = image_path_mapping[filename]
        else:
            if verbose:
                tqdm.write(f"Warning: Image {filename} not found in source directory, skipping...")
            continue
        
        # load image
        try:
            image_as_pil = read_image_as_pil(image_path)
        except Exception as e:
            if verbose:
                tqdm.write(f"Error loading image {image_path}: {e}")
            continue
        
        if verbose:
            tqdm.write(f"Processing: {filename} (Size: {image_as_pil.size})")
        
        # convert ground truth annotations to object_prediction_list
        object_prediction_gt_list: List[ObjectPrediction] = []
        for coco_annotation in coco_image.annotations:
            coco_annotation_dict = coco_annotation.json
            category_name = coco_annotation.category_name
            full_shape = [coco_image.height, coco_image.width]
            
            try:
                object_prediction_gt = ObjectPrediction.from_coco_annotation_dict(
                    annotation_dict=coco_annotation_dict, 
                    category_name=category_name, 
                    full_shape=full_shape
                )
                object_prediction_gt_list.append(object_prediction_gt)
            except Exception as e:
                if verbose:
                    tqdm.write(f"Error processing annotation for {filename}: {e}")
                continue
        
        if verbose:
            tqdm.write(f"Found {len(object_prediction_gt_list)} ground truth annotations")
        
        # export ground truth visualization
        time_start = time.time()
        # Since we're now using just the filename, save directly to the visual_dir
        output_dir = str(visual_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Use green color for ground truth annotations
        color = (0, 255, 0)  # Green color for ground truth
        
        result = visualize_object_predictions(
            np.ascontiguousarray(image_as_pil),
            object_prediction_list=object_prediction_gt_list,
            rect_th=visual_bbox_thickness,
            text_size=visual_text_size,
            text_th=visual_text_thickness,
            color=color,
            hide_labels=visual_hide_labels,
            hide_conf=True,  # Ground truth doesn't have confidence scores
            output_dir=output_dir,
            file_name=filename_without_extension,
            export_format=visual_export_format,
        )
        
        time_end = time.time() - time_start
        durations_in_seconds["visualization"] += time_end
        processed_count += 1
        
        if verbose:
            tqdm.write(
                f"Visualization time: {time_end:.2f} ms"
            )
    
    # Print summary
    print(f"\n{'='*60}")
    print("GROUND TRUTH VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"COCO annotations loaded in: {durations_in_seconds['coco_load']:.2f} seconds")
    print(f"Images processed: {processed_count}/{len(coco.images)}")
    print(f"Total visualization time: {durations_in_seconds['visualization']:.2f} seconds")
    print(f"Results exported to: {save_dir}")
    print(f"Ground truth visuals saved to: {visual_dir}")
    
    if verbose == 1:
        # Count total annotations across all images
        total_annotations = sum(len(coco_image.annotations) for coco_image in coco.images)
        
        print(f"\nCOCO Dataset Information:")
        print(f"- Total images: {len(coco.images)}")
        print(f"- Total annotations: {total_annotations}")
        print(f"- Categories: {len(coco.categories)}")
        
        # Print category information
        print(f"\nCategory Information:")
        for category in coco.categories:
            category_count = 0
            for coco_image in coco.images:
                for annotation in coco_image.annotations:
                    if annotation.category_id == category.id:
                        category_count += 1
            print(f"- {category.name}: {category_count} annotations")
    
    if return_dict:
        return {"export_dir": save_dir, "processed_count": processed_count}
    else:
        return {"export_dir": save_dir, "processed_count": processed_count}


def visualize_ground_truth_single_image(
    image_path: str,
    coco_json_path: str,
    output_path: str = None,
    visual_bbox_thickness: int = 2,
    visual_text_size: float = 1.0,
    visual_text_thickness: int = 2,
    visual_hide_labels: bool = False,
    visual_export_format: str = "png",
    verbose: bool = True,
):
    """
    Visualizes ground truth bounding boxes for a single image using COCO annotations.
    
    Args:
        image_path: str
            Path to the image file
        coco_json_path: str
            Path to COCO annotation JSON file
        output_path: str
            Output path for the visualized image. If None, will not save.
        visual_bbox_thickness: int
            Thickness of bounding box lines.
        visual_text_size: float
            Size of text labels.
        visual_text_thickness: int
            Thickness of text labels.
        visual_hide_labels: bool
            Hide class labels on bounding boxes.
        visual_export_format: str
            Export format for images ('jpg' or 'png').
        verbose: bool
            Print processing information.
    """
    
    # Load COCO annotations
    try:
        coco: Coco = Coco.from_coco_dict_or_path(coco_json_path)
        if verbose:
            print(f"Loaded COCO annotations with {len(coco.images)} images and {len(coco.categories)} categories")
    except Exception as e:
        print(f"Error loading COCO annotations from {coco_json_path}: {e}")
        return None
    
    # Extract image filename from path
    image_filename = Path(image_path).name
    
    # Find the corresponding COCO image entry
    coco_image = None
    for img in coco.images:
        # Compare filenames (handle both full paths and just filenames)
        if Path(img.file_name).name == image_filename:
            coco_image = img
            break
    
    if coco_image is None:
        print(f"Image {image_filename} not found in COCO annotations")
        return None
    
    if verbose:
        print(f"Found COCO image entry: {coco_image.file_name}")
        print(f"Number of annotations for this image: {len(coco_image.annotations)}")
    
    # Load image
    try:
        image_as_pil = read_image_as_pil(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    if verbose:
        print(f"Processing image: {image_path}")
        print(f"Image size: {image_as_pil.size}")
    
    # Convert annotations to object_prediction_list
    object_prediction_gt_list: List[ObjectPrediction] = []
    full_shape = [coco_image.height, coco_image.width]
    
    for coco_annotation in coco_image.annotations:
        try:
            coco_annotation_dict = coco_annotation.json
            category_name = coco_annotation.category_name
            
            object_prediction_gt = ObjectPrediction.from_coco_annotation_dict(
                annotation_dict=coco_annotation_dict, 
                category_name=category_name, 
                full_shape=full_shape
            )
            object_prediction_gt_list.append(object_prediction_gt)
        except Exception as e:
            if verbose:
                print(f"Error processing annotation: {e}")
            continue
    
    if verbose:
        print(f"Successfully processed {len(object_prediction_gt_list)} annotations")
    
    # Visualize ground truth
    color = (0, 255, 0)  # Green color for ground truth
    
    result = visualize_object_predictions(
        np.ascontiguousarray(image_as_pil),
        object_prediction_list=object_prediction_gt_list,
        rect_th=visual_bbox_thickness,
        text_size=visual_text_size,
        text_th=visual_text_thickness,
        color=color,
        hide_labels=visual_hide_labels,
        hide_conf=True,  # Ground truth doesn't have confidence scores
        output_dir=None,
        file_name=None,
        export_format=None,
    )
    
    # Save if output path is provided
    if output_path:
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to PIL and save
            result_image = Image.fromarray(result["image"])
            result_image.save(output_path)
            
            if verbose:
                print(f"Ground truth visualization saved to: {output_path}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    return result["image"]
    

def main():
    """
    Example usage of ground truth visualization functions.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ground truth bounding boxes from COCO annotations")
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to COCO annotation file")
    parser.add_argument("--source", type=str, default=None, help="Directory containing images (optional)")
    parser.add_argument("--project", type=str, default="runs/ground_truth", help="Output project directory")
    parser.add_argument("--name", type=str, default="exp", help="Output experiment name")
    parser.add_argument("--bbox_thickness", type=int, default=2, help="Bounding box line thickness")
    parser.add_argument("--text_size", type=float, default=1.0, help="Text label size")
    parser.add_argument("--text_thickness", type=int, default=2, help="Text label thickness")
    parser.add_argument("--hide_labels", action="store_true", help="Hide class labels")
    parser.add_argument("--export_format", type=str, default="png", choices=["png", "jpg"], help="Export format")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0, 1)")
    
    args = parser.parse_args()
    
    # Run ground truth visualization
    visualize_ground_truth_only(
        dataset_json_path=args.dataset_json,
        source=args.source,
        project=args.project,
        name=args.name,
        visual_bbox_thickness=args.bbox_thickness,
        visual_text_size=args.text_size,
        visual_text_thickness=args.text_thickness,
        visual_hide_labels=args.hide_labels,
        visual_export_format=args.export_format,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
