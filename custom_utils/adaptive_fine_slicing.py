import os
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, cpu_count
import torch
from torchvision.ops import nms
from sahi.prediction import get_prediction, get_sliced_prediction
from sahi.utils.file import save_json, increment_path
from sahi.utils.cv import read_image
from sahi.visualization import visualize_object_predictions


def get_dynamic_slice_parameters(object_density, base_slice_size):
    if object_density > 30:
        return base_slice_size // 2, base_slice_size // 2, 0.4, 0.4
    elif object_density > 15:
        return base_slice_size // 2, base_slice_size // 2, 0.3, 0.3
    elif object_density > 5:
        return base_slice_size, base_slice_size, 0.2, 0.2
    else:
        return None


def nms_merge(predictions, iou_threshold=0.5):
    if not predictions:
        return []

    boxes = torch.tensor([p.bbox.to_xyxy() for p in predictions])
    scores = torch.tensor([p.score.value for p in predictions])
    keep_indices = nms(boxes, scores, iou_threshold).tolist()
    return [predictions[i] for i in keep_indices]


def process_image_fine_slicing(args):
    filename, input_folder, detection_model, base_slice_size, data, save_dir, vis_params = args
    image_path = os.path.join(input_folder, filename)
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_h, image_w = image_np.shape[:2]
    filename_wo_ext = Path(filename).stem
    img_id = next((img["id"] for img in data.get("images", []) if img["file_name"].startswith(filename_wo_ext)), None)

    all_object_predictions = []
    total_time = 0

    # Split image into 2x2 grid
    grid_h, grid_w = image_h // 2, image_w // 2

    for row in range(2):
        for col in range(2):
            x1, y1 = col * grid_w, row * grid_h
            x2, y2 = min(x1 + grid_w, image_w), min(y1 + grid_h, image_h)
            sub_img = image_pil.crop((x1, y1, x2, y2))

            # Base prediction on the sub-image
            base_pred = get_prediction(sub_img, detection_model)
            object_density = len(base_pred.object_prediction_list)

            slice_params = get_dynamic_slice_parameters(object_density, base_slice_size)

            if slice_params:
                slice_width, slice_height, overlap_w, overlap_h = slice_params
                sliced_pred = get_sliced_prediction(
                    sub_img,
                    detection_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_h,
                    overlap_width_ratio=overlap_w,
                    postprocess_type="NMS",
                    postprocess_match_metric="IOU",
                    postprocess_match_threshold=0.5,
                    postprocess_min_area=32,
                    verbose=0
                )
                preds = sliced_pred.object_prediction_list
            else:
                preds = base_pred.object_prediction_list

            # Offset predictions to original image coordinates
            for pred in preds:
                pred.shift(x_offset=x1, y_offset=y1)
                all_object_predictions.append(pred)

    # Apply NMS to merged predictions
    merged_preds = nms_merge(all_object_predictions, iou_threshold=0.5)

    # Visualization
    visualize_object_predictions(
        image=np.ascontiguousarray(image_pil),
        object_prediction_list=merged_preds,
        rect_th=vis_params["bbox_thickness"],
        text_size=vis_params["text_size"],
        text_th=vis_params["text_thickness"],
        hide_labels=vis_params["hide_labels"],
        hide_conf=vis_params["hide_conf"],
        output_dir=save_dir,
        file_name=filename_wo_ext,
        export_format=vis_params["format"]
    )

    # COCO conversion
    coco_preds = [p.to_coco_dict(image_id=img_id) for p in merged_preds]
    return coco_preds


def predict_fine_sliced_images(input_folder, dataset_json_path, detection_model, base_slice_size=512):
    name = "exp"
    save_dir = Path(increment_path(Path("sliced_predictions") / name, exist_ok=False))
    os.makedirs(save_dir, exist_ok=True)

    data = {}
    if dataset_json_path:
        with open(dataset_json_path, "r") as file:
            data = json.load(file)

    vis_params = {
        "bbox_thickness": 2,
        "text_size": 0.5,
        "text_thickness": 1,
        "hide_labels": False,
        "hide_conf": False,
        "format": "png"
    }

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    tasks = [
        (filename, input_folder, detection_model, base_slice_size, data, save_dir, vis_params)
        for filename in image_files
    ]

    print(f"\n🚀 Running fine slicing prediction on {len(image_files)} images...")

    all_coco_preds = []
    with Pool(processes=min(cpu_count(), 4)) as pool:
        for preds in pool.imap_unordered(process_image_fine_slicing, tasks):
            all_coco_preds.extend(preds)

    if dataset_json_path:
        save_json(all_coco_preds, str(save_dir / "result.json"))
        print(f"\n📦 Saved COCO results to {save_dir / 'result.json'}")

    print(f"\n✅ Completed {len(image_files)} images.")
    return all_coco_preds
