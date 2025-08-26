import numpy as np
import torch
from typing import List, Tuple, Dict, Any
import copy

def calculate_iou_tensor(boxes1, boxes2):
    """Calculate IoU between two sets of boxes using PyTorch tensors"""
    # boxes format: [x1, y1, x2, y2]
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)

def calculate_distance_tensor(boxes1, boxes2):
    """Calculate Euclidean distance between centers of boxes"""
    center1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
    center1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
    center2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
    center2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    return torch.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

def is_ebb_tensor(box, score, tad=100.0, trd=0.1):
    """Determine if a detection is an Enhanced Bounding Box (EBB)"""
    area = (box[2] - box[0]) * (box[3] - box[1])
    aspect_ratio = (box[2] - box[0]) / (box[3] - box[1] + 1e-6)
    
    return (area > tad and aspect_ratio > trd and score > 0.5)

def is_s1_condition(ebb_box, det_box, ebb_score, det_score, td=0.3, ts=50.0):
    """Check if S1 condition is met"""
    iou = calculate_iou_tensor(ebb_box.unsqueeze(0), det_box.unsqueeze(0))[0]
    distance = calculate_distance_tensor(ebb_box.unsqueeze(0), det_box.unsqueeze(0))[0]
    
    return iou > td and distance < ts

def is_s2_condition(ebb_box, det_box, ebb_score, det_score, td1=0.1, td2=100.0):
    """Check if S2 condition is met"""
    iou = calculate_iou_tensor(ebb_box.unsqueeze(0), det_box.unsqueeze(0))[0]
    distance = calculate_distance_tensor(ebb_box.unsqueeze(0), det_box.unsqueeze(0))[0]
    
    return iou > td1 and distance < td2

def create_seb_tensor(ebb_box, det_box, ebb_score, det_score):
    """Create a SEB (Synthetic Enhanced Box) from EBB and det"""
    x1 = torch.min(ebb_box[0], det_box[0])
    y1 = torch.min(ebb_box[1], det_box[1])
    x2 = torch.max(ebb_box[2], det_box[2])
    y2 = torch.max(ebb_box[3], det_box[3])
    
    return torch.tensor([x1, y1, x2, y2]), max(ebb_score, det_score)

def standard_nms_tensor(boxes, scores, iou_threshold=0.5):
    """Standard Non-Maximum Suppression using PyTorch tensors"""
    if len(boxes) == 0:
        return torch.empty(0, 4), torch.empty(0)
    
    # Sort by score in descending order
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    sorted_boxes = boxes[sorted_indices]
    
    keep_indices = []
    while len(sorted_boxes) > 0:
        # Take the highest scoring detection
        current_box = sorted_boxes[0]
        keep_indices.append(sorted_indices[0].item())
        
        # Remove current box
        sorted_boxes = sorted_boxes[1:]
        sorted_indices = sorted_indices[1:]
        
        if len(sorted_boxes) == 0:
            break
        
        # Calculate IoU with remaining boxes
        ious = calculate_iou_tensor(current_box.unsqueeze(0), sorted_boxes)[0]
        
        # Keep boxes with IoU <= threshold
        keep_mask = ious <= iou_threshold
        sorted_boxes = sorted_boxes[keep_mask]
        sorted_indices = sorted_indices[keep_mask]
    
    if keep_indices:
        return boxes[keep_indices], scores[keep_indices]
    else:
        return torch.empty(0, 4), torch.empty(0)

def sfp_nms_tensor(boxes, scores, labels,
                   tad=100.0, trd=0.1, td=0.3, ts=50.0,
                   td1=0.1, td2=100.0, nms_iou_threshold=0.5):
    """
    SFP-NMS using PyTorch tensors
    Args:
        boxes: tensor of shape [N, 4] with format [x1, y1, x2, y2]
        scores: tensor of shape [N]
        labels: tensor of shape [N]
    Returns:
        filtered_boxes, filtered_scores, filtered_labels
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Make copies to avoid modifying original tensors
    boxes = boxes.clone()
    scores = scores.clone()
    labels = labels.clone()
    
    # Step 1: Identify EBBs (Enhanced Bounding Boxes)
    ebb_indices = []
    for i in range(len(boxes)):
        if is_ebb_tensor(boxes[i], scores[i], tad, trd):
            ebb_indices.append(i)
    
    # Step 2: Process EBBs and detections
    boxes_to_remove = set()
    new_boxes = []
    new_scores = []
    new_labels = []
    
    for ebb_idx in ebb_indices:
        ebb_box = boxes[ebb_idx]
        ebb_score = scores[ebb_idx]
        ebb_label = labels[ebb_idx]
        
        for det_idx in range(len(boxes)):
            if det_idx == ebb_idx or det_idx in boxes_to_remove:
                continue
                
            det_box = boxes[det_idx]
            det_score = scores[det_idx]
            det_label = labels[det_idx]
            
            # Only process if labels match and IoU > 0
            if det_label == ebb_label:
                iou = calculate_iou_tensor(ebb_box.unsqueeze(0), det_box.unsqueeze(0))[0]
                
                if iou > 0:
                    if is_s1_condition(ebb_box, det_box, ebb_score, det_score, td, ts):
                        # S1 Processing: enhance score and remove EBB
                        scores[det_idx] = max(det_score, ebb_score)
                        boxes_to_remove.add(ebb_idx)
                        break
                        
                    elif is_s2_condition(ebb_box, det_box, ebb_score, det_score, td1, td2):
                        # S2 Processing: create SEB and remove both
                        seb_box, seb_score = create_seb_tensor(ebb_box, det_box, ebb_score, det_score)
                        new_boxes.append(seb_box)
                        new_scores.append(seb_score)
                        new_labels.append(ebb_label)
                        boxes_to_remove.add(ebb_idx)
                        boxes_to_remove.add(det_idx)
                        break
    
    # Step 3: Remove processed boxes and add new SEBs
    keep_mask = torch.ones(len(boxes), dtype=torch.bool)
    for idx in boxes_to_remove:
        keep_mask[idx] = False
    
    filtered_boxes = boxes[keep_mask]
    filtered_scores = scores[keep_mask]
    filtered_labels = labels[keep_mask]
    
    # Add new SEBs
    if new_boxes:
        new_boxes_tensor = torch.stack(new_boxes)
        new_scores_tensor = torch.tensor(new_scores)
        new_labels_tensor = torch.tensor(new_labels)
        
        filtered_boxes = torch.cat([filtered_boxes, new_boxes_tensor], dim=0)
        filtered_scores = torch.cat([filtered_scores, new_scores_tensor], dim=0)
        filtered_labels = torch.cat([filtered_labels, new_labels_tensor], dim=0)
    
    # Step 4: Apply final standard NMS
    if len(filtered_boxes) > 0:
        final_boxes, final_scores = standard_nms_tensor(filtered_boxes, filtered_scores, nms_iou_threshold)
        final_labels = filtered_labels[:len(final_boxes)]  # Labels are preserved in order
        return final_boxes, final_scores, final_labels
    
    return filtered_boxes, filtered_scores, filtered_labels

# Apply SFP-NMS (Synthetic Feature Pyramid Non-Maximum Suppression)
def apply_sfp_nms(detections, 
                  tad=100.0, trd=0.1, td=0.3, ts=50.0,
                  td1=0.1, td2=100.0, nms_iou_threshold=0.5):
    """
    Apply SFP-NMS to detections
    Args:
        detections: List of detection objects with bbox, score, and category attributes
        tad: Area threshold for EBB detection
        trd: Ratio threshold for EBB detection
        td: IoU threshold for S1 condition
        ts: Distance threshold for S1 condition
        td1: IoU threshold for S2 condition
        td2: Distance threshold for S2 condition
        nms_iou_threshold: Final NMS IoU threshold
    Returns:
        List of filtered detections in the same format as input
    """
    if not detections:
        return []
    
    # Extract boxes, scores, and labels from detections
    boxes, scores, labels = [], [], []
    for det in detections:
        boxes.append(det.bbox.to_xyxy())
        scores.append(det.score.value)
        labels.append(det.category.id)
    
    # Convert to tensors
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels)
    
    # Apply SFP-NMS per class
    keep_indices = []
    unique_labels = labels.unique()
    
    for label in unique_labels:
        # Get indices for current class
        idxs = (labels == label).nonzero(as_tuple=False).squeeze(1)
        
        if len(idxs) == 0:
            continue
            
        # Get boxes, scores for current class
        class_boxes = boxes[idxs]
        class_scores = scores[idxs]
        class_labels = labels[idxs]
        
        # Apply SFP-NMS for this class
        filtered_boxes, filtered_scores, filtered_labels = sfp_nms_tensor(
            class_boxes, class_scores, class_labels,
            tad, trd, td, ts, td1, td2, nms_iou_threshold
        )
        
        # Map back to original indices or create new detections for SEBs
        for i in range(len(filtered_boxes)):
            # Try to find the original detection that matches this filtered result
            found_original = False
            for j, orig_idx in enumerate(idxs):
                if (torch.allclose(boxes[orig_idx], filtered_boxes[i], atol=1e-6) and
                    torch.allclose(scores[orig_idx], filtered_scores[i], atol=1e-6)):
                    keep_indices.append(orig_idx.item())
                    found_original = True
                    break
            
            # If not found in original, it's a new SEB - create a new detection
            if not found_original:
                # Create a new detection object for the SEB
                # We'll need to create a mock detection that maintains the same interface
                seb_detection = create_seb_detection(filtered_boxes[i], filtered_scores[i], filtered_labels[i])
                detections.append(seb_detection)
                keep_indices.append(len(detections) - 1)
    
    # Return filtered detections
    return [detections[i] for i in keep_indices]

def create_seb_detection(box, score, label):
    """Create a new detection object for SEB"""
    # This is a placeholder - you may need to adapt this based on your detection class
    class MockBbox:
        def __init__(self, box_tensor):
            self.box = box_tensor
        def to_xyxy(self):
            return self.box.tolist()
    
    class MockScore:
        def __init__(self, score_value):
            self.value = score_value
    
    class MockCategory:
        def __init__(self, label_id):
            self.id = label_id
    
    class MockDetection:
        def __init__(self, bbox, score, category):
            self.bbox = bbox
            self.score = score
            self.category = category
    
    return MockDetection(MockBbox(box), MockScore(score), MockCategory(label))