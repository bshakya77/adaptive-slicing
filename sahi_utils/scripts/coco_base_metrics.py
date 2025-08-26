"""
Utility functions for computing true positives (TP), false positives (FP) and
false negatives (FN) using pycocotools.  This module provides a high‑level
interface to load COCO‑formatted ground truth and detection results, run the
COCO evaluation pipeline and extract TP/FP/FN counts from the per‑image
evaluation results (`COCOeval.evalImgs`).

Example usage from the command line:

    python coco_tp_fp_fn.py path/to/instances_val2017.json path/to/detections.json

or from within Python:

    from coco_tp_fp_fn import compute_tp_fp_fn
    iou_thrs, tp, fp, fn = compute_tp_fp_fn(gt_json, dt_json)
    for t, threshold in enumerate(iou_thrs):
        print(f"IoU {threshold:.2f}: TP={tp[t]}, FP={fp[t]}, FN={fn[t]}")

The evaluation follows the logic described in the COCO API: detections are
matched to ground truths in descending order of confidence.  A detection is
considered a true positive if it matches a ground truth of the same class
with intersection‑over‑union (IoU) above the threshold and neither the
detection nor the ground truth is flagged as ignored.  Unmatched detections
count as false positives, and unmatched ground truths count as false
negatives.
"""

import argparse
from typing import List, Tuple

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pycocotools is required for computing TP/FP/FN. Please install it via `pip install pycocotools`."
    ) from exc


def compute_tp_fp_fn(
    dataset_json_path: str,
    result_json_path: str,
    iou_type: str = "bbox",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute true positives, false positives and false negatives for a COCO dataset.

    This function loads the ground‑truth annotations and detection results,
    runs COCO evaluation and then iterates over the per‑image evaluation
    results to count the number of true positives (TP), false positives (FP)
    and false negatives (FN) for each intersection‑over‑union threshold.

    Args:
        dataset_json_path (str): Path to a COCO dataset JSON file containing
            annotations (e.g. instances_val2017.json).
        result_json_path (str): Path to a COCO results JSON file containing
            detections (must have the same format as returned by the model).
        iou_type (str): Type of evaluation.  Must be 'bbox' or 'segm'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - ``iou_thrs``: 1D array of IoU thresholds used in evaluation.
            - ``tp``: 1D array of true positive counts for each IoU threshold.
            - ``fp``: 1D array of false positive counts for each IoU threshold.
            - ``fn``: 1D array of false negative counts for each IoU threshold.

    Notes:
        * The counts returned are aggregated across all images, categories and
          area ranges defined by the COCO evaluation parameters.  If you want
          per‑class or per‑area breakdowns, further filtering of ``evalImgs``
          is required.
        * The default IoU thresholds follow the COCO convention, ranging from
          0.50 to 0.95 in steps of 0.05.  You can modify
          ``COCOeval.params.iouThrs`` before calling ``evaluate()`` to use
          custom thresholds.
    """
    # Load ground truth and detection results
    coco_gt = COCO(dataset_json_path)
    coco_dt = coco_gt.loadRes(result_json_path)

    # Configure evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    # Run evaluation and accumulate precision/recall
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Number of IoU thresholds
    iou_thrs = coco_eval.params.iouThrs
    T = len(iou_thrs)

    # Initialise counters for each threshold
    tp = np.zeros(T, dtype=int)
    fp = np.zeros(T, dtype=int)
    fn = np.zeros(T, dtype=int)

    # Iterate over each per‑image evaluation result
    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue
        # Extract match/ignore arrays
        dt_matches = eval_img["dtMatches"]  # shape [T, maxDet]
        dt_ignore = eval_img["dtIgnore"]    # shape [T, maxDet], booleans
        gt_matches = eval_img["gtMatches"]  # shape [T, numGt]
        gt_ignore = eval_img["gtIgnore"]    # shape [T, numGt], booleans

        # Number of detections and ground truths for this eval
        # dt_matches.shape[1] corresponds to maxDets for this category/area
        # gt_matches.shape[1] corresponds to number of ground truths
        for t_idx in range(T):
            # Count TP and FP for detections
            for d_idx in range(dt_matches.shape[1]):
                # dtIgnore can be 1‑D or 2‑D; check ndim to decide
                ignore_flag = (
                    dt_ignore[t_idx, d_idx]  # 2‑D: (T, maxDet)
                    if getattr(dt_ignore, "ndim", 2) == 2
                    else dt_ignore[d_idx]    # 1‑D: (maxDet,)
                )
                if ignore_flag:
                    continue
                tp[t_idx] += 1 if dt_matches[t_idx, d_idx] > 0 else 0
                fp[t_idx] += 0 if dt_matches[t_idx, d_idx] > 0 else 1

            # Count FN for ground truths
            for g_idx in range(gt_matches.shape[1]):
                ignore_flag = (
                    gt_ignore[t_idx, g_idx]  # 2‑D: (T, numGt)
                    if getattr(gt_ignore, "ndim", 2) == 2
                    else gt_ignore[g_idx]    # 1‑D: (numGt,)
                )
                if ignore_flag:
                    continue
                if gt_matches[t_idx, g_idx] == 0:
                    fn[t_idx] += 1

    return iou_thrs, tp, fp, fn


def main() -> None:
    """Entry point for command‑line usage.  Parses arguments and prints counts."""
    parser = argparse.ArgumentParser(
        description="Compute TP/FP/FN using the COCO evaluation API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_json_path",
        help="Path to COCO annotation file (ground truth).",
    )
    parser.add_argument(
        "result_json_path",
        help="Path to COCO results file (predictions).",
    )
    parser.add_argument(
        "--iou_type",
        default="bbox",
        choices=["bbox", "segm"],
        help="Evaluation type (bounding boxes or segmentation)",
    )
    args = parser.parse_args()

    iou_thrs, tp, fp, fn = compute_tp_fp_fn(
        args.dataset_json_path,
        args.result_json_path,
        args.iou_type,
    )

    # Print results
    print("******************************************************************")
    for i, thr in enumerate(iou_thrs):
        print(
            f"IoU threshold {thr:.2f}: TP={tp[i]}, FP={fp[i]}, FN={fn[i]}"
        )
    print("******************************************************************")
    

if __name__ == "__main__":
    main()