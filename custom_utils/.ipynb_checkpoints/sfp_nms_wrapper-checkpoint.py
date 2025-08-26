"""
Wrapper around the SFP‑NMS algorithm to integrate it with detection
pipelines that currently use the standard NMS implementation.

The baseline `apply_nms` function in `NMS.py` expects a list of detection
objects and returns a filtered list after applying standard non‑maximum
suppression (NMS).  Each detection exposes a `bbox` object with a
`to_xyxy()` method (returning `[x1, y1, x2, y2]`), a `score` object with
a `value` attribute (the confidence score as a float), and a `category`
object with an `id` attribute (the class label as an integer).  This wrapper
provides an `apply_sfp_nms` function that takes the same input, calls into
the SFP‑NMS algorithm defined in `SFP_NMS.py`, and returns a list of
detections in the same format.  Internally it converts detections into
`BoundingBox` objects, runs SFP‑NMS per class label, and then builds new
detection objects with updated coordinates and scores.

This allows users to swap out `apply_nms` for `apply_sfp_nms` in their
existing code without further modification.
"""
from __future__ import annotations

from typing import List, Any
import copy

from custom_utils.SFP_NMS import BoundingBox, sfp_nms  # type: ignore


def apply_sfp_nms(
    detections: List[Any],
    tad: float = 100.0,
    trd: float = 0.1,
    td: float = 0.3,
    ts: float = 50.0,
    td1: float = 0.1,
    td2: float = 100.0,
    nms_iou_threshold: float = 0.5,
) -> List[Any]:
    """
    Apply the SFP‑NMS algorithm to a list of detection objects.

    This function mirrors the signature of `apply_nms` from `NMS.py` but
    internally uses the more sophisticated SFP‑NMS algorithm defined in
    `SFP_NMS.py`.  It groups detections by their class labels, converts each
    group into `BoundingBox` objects, runs SFP‑NMS with the given parameters,
    and then rebuilds new detection objects (deep copies of the originals)
    with updated bounding boxes and scores.  The intent is that the calling
    code can swap the call to `apply_nms` with `apply_sfp_nms` without
    having to change the rest of the pipeline.

    Args:
        detections: List of detection objects.  Each detection must expose
            `bbox`, `score` and `category` attributes with the same semantics
            as those expected by `apply_nms` (see above).  The function
            makes deep copies of these detections and does not modify the
            originals.
        tad, trd, td, ts, td1, td2: SFP‑NMS specific thresholds controlling
            EBB detection and merging behaviour.  Defaults mirror the values
            used in the standalone `sfp_nms` implementation.
        nms_iou_threshold: IoU threshold used during the final NMS stage.

    Returns:
        A list of detection objects (of the same type as the input) after
        applying SFP‑NMS.  Each detection in the returned list corresponds
        to one of the `BoundingBox` objects produced by SFP‑NMS and is a
        deep copy of one of the original detections with updated bounding
        box coordinates and confidence score.
    """
    if not detections:
        return []

    # Determine the unique class labels present in the input.  We mirror the
    # behaviour of the baseline NMS, which performs suppression per class.
    labels = [getattr(det.category, "id") for det in detections]
    unique_labels = sorted(set(labels))

    results: List[Any] = []

    for label in unique_labels:
        # Collect detections belonging to the current class label
        class_dets = [det for det in detections if getattr(det.category, "id") == label]
        if not class_dets:
            continue

        # Convert to internal BoundingBox objects for SFP‑NMS
        boxes: List[BoundingBox] = []
        for det in class_dets:
            # Expect bbox.to_xyxy() to return a sequence of four numbers
            coords = det.bbox.to_xyxy()
            try:
                x1, y1, x2, y2 = [float(c) for c in coords]
            except Exception:
                # Coerce to tuple if necessary
                coords = tuple(coords)
                x1, y1, x2, y2 = [float(c) for c in coords]
            score_val = float(getattr(det.score, "value"))
            boxes.append(BoundingBox(x1, y1, x2, y2, score_val, class_id=label))

        # Run SFP‑NMS on this class
        kept_boxes = sfp_nms(
            boxes,
            tad=tad,
            trd=trd,
            td=td,
            ts=ts,
            td1=td1,
            td2=td2,
            nms_iou_threshold=nms_iou_threshold,
        )

        # Reconstruct detection objects from the kept BoundingBox objects
        # Use the first detection in the class as a prototype for deep copying
        prototype = class_dets[0]
        for bb in kept_boxes:
            # Deep copy to preserve other attributes
            new_det = copy.deepcopy(prototype)
            # Attempt to update the bbox.  We assume that the bbox object can
            # be re‑constructed by calling its class with x1, y1, x2, y2.
            try:
                bbox_class = prototype.bbox.__class__
                new_det.bbox = bbox_class(bb.x1, bb.y1, bb.x2, bb.y2)
            except Exception:
                # If the bbox is not directly constructible, assign a simple
                # object with a `to_xyxy` method and coordinate attributes.
                class SimpleBBox:
                    def __init__(self, x1, y1, x2, y2):
                        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
                    def to_xyxy(self):
                        return (self.x1, self.y1, self.x2, self.y2)
                new_det.bbox = SimpleBBox(bb.x1, bb.y1, bb.x2, bb.y2)
            # Update the score; if `score` has a `value` attribute, update it
            if hasattr(new_det.score, "value"):
                try:
                    # If score is a wrapper (e.g. a tensor), set its `.value`
                    setattr(new_det.score, "value", bb.score)
                except Exception:
                    # Otherwise, replace the score object entirely
                    try:
                        new_det.score = type(prototype.score)(bb.score)
                    except Exception:
                        new_det.score = bb.score
            else:
                # Score attribute is a plain float
                new_det.score = bb.score
            # Update category id to be explicit (though it should already match)
            if hasattr(new_det.category, "id"):
                try:
                    setattr(new_det.category, "id", bb.class_id)
                except Exception:
                    # If category is immutable, create a new one if possible
                    try:
                        new_det.category = type(prototype.category)(bb.class_id)
                    except Exception:
                        pass
            results.append(new_det)

    return results