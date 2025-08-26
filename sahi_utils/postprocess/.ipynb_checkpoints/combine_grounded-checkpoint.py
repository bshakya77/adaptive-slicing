# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2021.

import logging
from typing import List

import torch
from sahi.postprocess.utils import ObjectPredictionList, has_match, merge_object_prediction_pair
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)

def batched_nms(predictions: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    scores = predictions[:, 4].squeeze()
    category_ids = predictions[:, 5].squeeze()
    keep_mask = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_indices = nms(predictions[curr_indices], match_metric, match_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    # sort selected indices by their scores
    keep_indices = keep_indices[scores[keep_indices].sort(descending=True)[1]].tolist()
    return keep_indices


def nms(
    predictions: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    # we extract coordinates for every
    # prediction box present in P
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]

    # we extract the confidence scores as well
    scores = predictions[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
    
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoU of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU less than thresh_iou
        mask = match_metric_value < match_threshold
        order = order[mask]
    print("Total Valid prediction: ", len(keep))
    return keep


def batched_greedy_nmm(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply greedy version of non-maximum merging per category to avoid detecting
    too many overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = greedy_nmm(object_predictions_as_tensor[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()
        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [curr_indices_list[curr_merge_ind] for curr_merge_ind in curr_merge_list]
            keep_to_merge_list[keep] = merge_list
    return keep_to_merge_list


def greedy_nmm(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply greedy version of non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        object_predictions_as_list: ObjectPredictionList Object prediction objects
            to be merged.
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    keep_to_merge_list = {}

    # we extract coordinates for every
    # prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]

    # we extract the confidence scores as well
    scores = object_predictions_as_tensor[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

         
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            keep_to_merge_list[idx.tolist()] = []
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU/IoS less than thresh_iou
        mask = match_metric_value < match_threshold
        matched_box_indices = order[(mask == False).nonzero().flatten()].flip(dims=(0,))
        unmatched_indices = order[(mask == True).nonzero().flatten()]

        # update box pool
        order = unmatched_indices[scores[unmatched_indices].argsort()]

        # create keep_ind to merge_ind_list mapping
        keep_to_merge_list[idx.tolist()] = []

        for matched_box_ind in matched_box_indices.tolist():
            keep_to_merge_list[idx.tolist()].append(matched_box_ind)

    return keep_to_merge_list


def batched_nmm(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum merging per category to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = nmm(object_predictions_as_tensor[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()
        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [curr_indices_list[curr_merge_ind] for curr_merge_ind in curr_merge_list]
            keep_to_merge_list[keep] = merge_list
    return keep_to_merge_list


def nmm(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        object_predictions_as_list: ObjectPredictionList Object prediction objects
            to be merged.
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    keep_to_merge_list = {}
    merge_to_keep = {}

    # we extract coordinates for every
    # prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]

    # we extract the confidence scores as well
    scores = object_predictions_as_tensor[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort(descending=True)

    for ind in range(len(object_predictions_as_tensor)):
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        pred_ind = order[ind]
        pred_ind = pred_ind.tolist()

        # remove selected pred
        other_pred_inds = order[order != pred_ind]

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=other_pred_inds)
        xx2 = torch.index_select(x2, dim=0, index=other_pred_inds)
        yy1 = torch.index_select(y1, dim=0, index=other_pred_inds)
        yy2 = torch.index_select(y2, dim=0, index=other_pred_inds)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[pred_ind])
        yy1 = torch.max(yy1, y1[pred_ind])
        xx2 = torch.min(xx2, x2[pred_ind])
        yy2 = torch.min(yy2, y2[pred_ind])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=other_pred_inds)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[pred_ind]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[pred_ind])
            # find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU/IoS less than thresh_iou
        mask = match_metric_value < match_threshold
        matched_box_indices = other_pred_inds[(mask == False).nonzero().flatten()].flip(dims=(0,))

        # create keep_ind to merge_ind_list mapping
        if pred_ind not in merge_to_keep:
            keep_to_merge_list[pred_ind] = []

            for matched_box_ind in matched_box_indices.tolist():
                if matched_box_ind not in merge_to_keep:
                    keep_to_merge_list[pred_ind].append(matched_box_ind)
                    merge_to_keep[matched_box_ind] = pred_ind

        else:
            keep = merge_to_keep[pred_ind]
            for matched_box_ind in matched_box_indices.tolist():
                if matched_box_ind not in keep_to_merge_list and matched_box_ind not in merge_to_keep:
                    keep_to_merge_list[keep].append(matched_box_ind)
                    merge_to_keep[matched_box_ind] = keep

    return keep_to_merge_list


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions"""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
        conf_threshold: float = 0.3,
        min_area: float = 64,
        image_height:int = None
        
    ):
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric
        self.conf_threshold = conf_threshold
        self.min_area = min_area
        self.image_height = image_height
        
        check_requirements(["torch"])

    def __call__(self):
        raise NotImplementedError()


#******************************************************************************************#


def spatial_context_filter(predictions: torch.tensor, image_height: int, threshold: float = 0.4):
    """
    Removes detections that appear too high in the image if they belong to grounded categories.

    Args:
        predictions: tensor of shape [num_boxes, 6] where columns are [x1, y1, x2, y2, score, category_id].
        image_height: height of the image.
        threshold: ratio of image height above which grounded objects are discarded.
        grounded_categories: set of category IDs considered grounded.

    Returns:
        Tensor of predictions after spatial filtering.
    """
    grounded_categories = {0,1,2,3,4,5,6,7,8,9}
    if grounded_categories is None:
        return predictions

    y_bottom = predictions[:, 3]
    category_ids = predictions[:, 5].long()
    #print("y_bottom", y_bottom)
    print("Alevation:", image_height * threshold)
    # Grounded mask checks if objects are grounded and correctly placed in the lower portion of the image
    #grounded_mask = torch.tensor([cat_id in grounded_categories for cat_id in category_ids])
    height_mask = y_bottom >= (image_height * threshold)

    keep_mask = height_mask
    print(keep_mask)
    return predictions[keep_mask]

def quick_overlap_predictions(predictions, iou_threshold=0.7):
    """
    Quickly filters out overlapping predictions based on Intersection over Union (IoU).

    Args:
        predictions (list[dict]): Each with 'bbox': [x, y, width, height] and 'score'.
        iou_threshold (float): IoU threshold above which predictions are considered overlapping.

    Returns:
        list[dict]: Filtered predictions.
    """
    predictions.sort(key=lambda x: x['score'], reverse=True)
    filtered_preds = []

    for pred in predictions:
        overlap = False
        for kept_pred in filtered_preds:
            if compute_iou(pred['bbox'], kept_pred['bbox']) > iou_threshold:
                overlap = True
                break
        if not overlap:
            filtered_preds.append(pred)

    return filtered_preds


#New helper function to filter low-confidence and small-area predictions.
def filter_predictions(predictions: torch.tensor, conf_threshold: float = 0.3, min_area: float = 16,
                      iou_min: float =0.4, iou_max: float = 0.7, area_threshold: int = 1024):
    """
    Filters out predictions with confidence scores below conf_threshold or with an area smaller than min_area.
    Args:
        predictions: tensor of shape [num_boxes, 6] where column 4 is the confidence.
        conf_threshold: Minimum confidence score required.
        min_area: Minimum area (width * height) required.
    Returns:
        Filtered predictions tensor.
    """
    scores = predictions[:, 4]
    conf_mask = scores >= conf_threshold

    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    area_mask = areas >= min_area

    # Adaptive IoU thresholding
    adaptive_iou = torch.where(areas < area_threshold, iou_max, iou_min)
    
    valid_mask = conf_mask & area_mask
    return predictions[valid_mask]

#New helper function to filter low-confidence and small-area predictions.
def adaptive_filter_predictions(predictions_org: torch.tensor, conf_threshold: float = 0.5, min_area: float = 64, iou_min: float =0.4, iou_max: float = 0.8, area_threshold: int = 1024, image_height: int = None):
    """
    Filters out predictions with confidence scores below conf_threshold or with an area smaller than min_area.
    Args:
        predictions: tensor of shape [num_boxes, 6] where column 4 is the confidence.
        conf_threshold: Minimum confidence score required.
        min_area: Minimum area (width * height) required.
    Returns:
        Filtered predictions tensor.
    """
    
    print("Image Height: ", image_height)
    predictions = spatial_context_filter(predictions_org, image_height, threshold=0.4)
    print("After Spatial Filter: ", len(predictions))
    
    scores = predictions[:, 4]
    conf_mask = scores >= conf_threshold

    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    area_mask = areas >= min_area
    
    valid_mask = conf_mask & area_mask
    filtered_areas = areas[valid_mask]
    
    adaptive_iou = torch.where(filtered_areas < area_threshold, iou_max, iou_min)
    
    return predictions[valid_mask], adaptive_iou

def optimized_nms_bk(predictions: torch.tensor, adaptive_iou: torch.float32, match_metric: str = "IOU", conf_threshold: float = 0.5, min_area: float = 64):
    """
    Optimized NMS that first filters out low-confidence and small-area predictions.
    Args:
        predictions: Tensor of shape [num_boxes, 6] (x1, y1, x2, y2, score, category_id).
        match_metric: "IOU" or "IOS".
        match_threshold: IoU/IOS threshold for suppression.
        conf_threshold: Confidence threshold to filter predictions.
        min_area: Minimum area required for a valid prediction.
    Returns:
        Tensor of kept predictions after applying optimized NMS.
    """
    # Store original indices before filtering
    #original_indices = torch.arange(len(predictions))

    # Apply pre-filtering (low confidence and small-area removal)
    #filtered_predictions, adaptive_iou = adaptive_filter_predictions(predictions, conf_threshold, min_area)

    #if filtered_predictions.numel() == 0:
    #    return torch.empty((0, 6))  # Return empty tensor if no detections remain
    
    # Map original indices to filtered predictions
    #filtered_indices = original_indices[:len(filtered_predictions)]

    x1, y1, x2, y2 = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    scores = predictions[:, 4]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by confidence score (descending order)
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        idx = order[0]  # Pick the highest confidence prediction
        keep.append(filtered_indices[idx].item())  # Store the original index
        order = order[1:]  # Remove the highest confidence box from the list

        if order.numel() == 0:
            break

        # Gather remaining boxes
        xx1 = torch.index_select(x1, 0, order)
        yy1 = torch.index_select(y1, 0, order)
        xx2 = torch.index_select(x2, 0, order)
        yy2 = torch.index_select(y2, 0, order)

        # Compute intersection coordinates
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # Compute width and height of intersection
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        rem_areas = torch.index_select(areas, 0, order)
        
        # Compute IoU or IOS
        if match_metric == "IOU":
            union = rem_areas + areas[idx] - inter
            match_metric_value = inter / union
        elif match_metric == "IOS":
            smaller_area = torch.min(rem_areas, areas[idx])
            match_metric_value = inter / smaller_area
            #print("match_metric_value_ios", match_metric_value)
        else:
            raise ValueError("Unsupported match_metric. Use 'IOU' or 'IOS'.")
            print("Error")
        #print("match_metric_value_iou", match_metric_value)
        #print("Order", order)
        # Get correct adaptive IoU for each comparison
        iou_thresholds = torch.index_select(adaptive_iou, 0, order)
        #print("adaptive_iou", iou_thresholds)
        # Keep boxes with IoU/IOS less than their respective adaptive threshold
        mask = match_metric_value < iou_thresholds
        order = order[mask]
        
    print("Final Bounding Box Count:", len(keep))
    # Return the final kept predictions
    return keep



def batched_optimized_nms(predictions: torch.tensor, adaptive_iou:torch.float32, match_metric: str = "IOU", conf_threshold: float = 0.5, min_area: float = 64):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    scores = predictions[:, 4].squeeze()
    category_ids = predictions[:, 5].squeeze()
    keep_mask = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_indices = optimized_nms(predictions[curr_indices], adaptive_iou, match_metric, conf_threshold, min_area)
        #print("+++++++++++++++++++++++++++++++++++++++++")
        #print(curr_indices[curr_keep_indices])
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    # sort selected indices by their scores
    keep_indices = keep_indices[scores[keep_indices].sort(descending=True)[1]].tolist()
    return keep_indices
    
def optimized_nms(predictions: torch.tensor,adaptive_iou:torch.float32, match_metric: str = "IOU", conf_threshold: float = 0.3, min_area: float = 64):
    """
    Optimized NMS that first filters out low-confidence and small-area predictions.
    Args:
        predictions: Tensor of shape [num_boxes, 6] (x1, y1, x2, y2, score, category_id).
        match_metric: "IOU" or "IOS".
        match_threshold: IoU/IOS threshold for suppression.
        conf_threshold: Confidence threshold to filter predictions.
        min_area: Minimum area required for a valid prediction.
    Returns:
        A list of indices for the kept predictions.
    """
    # Filter out low-quality predictions first
    #print("Prediction Count Original: ", len(predictions))
    #predictions = filter_predictions(predictions, conf_threshold, 64)
    print("Filtered Prediction: ", len(predictions))
    #predictions = filter_border_predictions(predictions, image_size, 5)
    
    if predictions.numel() == 0:
        return []
    
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort the predictions by their confidence scores (ascending order)
    order = scores.argsort()
    keep = []

    while order.numel() > 0:
        idx = order[-1]  # index of highest score prediction
        keep.append(idx.item())
        order = order[:-1]
        if order.numel() == 0:
            break

        # Gather remaining boxes
        xx1 = torch.index_select(x1, 0, order)
        yy1 = torch.index_select(y1, 0, order)
        xx2 = torch.index_select(x2, 0, order)
        yy2 = torch.index_select(y2, 0, order)

        # Compute intersection coordinates
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # Compute width and height of intersection
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        rem_areas = torch.index_select(areas, 0, order)
        if match_metric == "IOU":
            union = (rem_areas - inter) + areas[idx]
            match_metric_value = inter / union
        elif match_metric == "IOS":
            smaller = torch.min(rem_areas, areas[idx])
            match_metric_value = inter / smaller
        else:
            raise ValueError("Unsupported match_metric. Use 'IOU' or 'IOS'.")
  
        iou_thresholds = torch.index_select(adaptive_iou, 0, order)
        #print("adaptive_iou", iou_thresholds)
        # Keep boxes with IoU/IOS less than their respective adaptive threshold
        mask = match_metric_value < iou_thresholds
        # Keep boxes with IoU/IOS less than the threshold
        #mask = match_metric_value < match_threshold
        order = order[mask]
    print("Final Bounding Box Count:", len(keep))
    return keep


def optimized_nms_org(predictions: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5, conf_threshold: float = 0.3, min_area: float = 16):
    """
    Optimized NMS that first filters out low-confidence and small-area predictions.
    Args:
        predictions: Tensor of shape [num_boxes, 6] (x1, y1, x2, y2, score, category_id).
        match_metric: "IOU" or "IOS".
        match_threshold: IoU/IOS threshold for suppression.
        conf_threshold: Confidence threshold to filter predictions.
        min_area: Minimum area required for a valid prediction.
    Returns:
        A list of indices for the kept predictions.
    """
    # Filter out low-quality predictions first
    print("Prediction Count Original: ", len(predictions))
    predictions = filter_predictions(predictions, conf_threshold, 16)
    print("Filtered Prediction (low and min area elimination): ", len(predictions))
    #predictions = filter_border_predictions(predictions, image_size, 5)
    
    if predictions.numel() == 0:
        return []
    
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort the predictions by their confidence scores (ascending order)
    order = scores.argsort()
    keep = []

    while order.numel() > 0:
        idx = order[-1]  # index of highest score prediction
        keep.append(idx.item())
        order = order[:-1]
        if order.numel() == 0:
            break

        # Gather remaining boxes
        xx1 = torch.index_select(x1, 0, order)
        yy1 = torch.index_select(y1, 0, order)
        xx2 = torch.index_select(x2, 0, order)
        yy2 = torch.index_select(y2, 0, order)

        # Compute intersection coordinates
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # Compute width and height of intersection
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        rem_areas = torch.index_select(areas, 0, order)
        if match_metric == "IOU":
            union = (rem_areas - inter) + areas[idx]
            match_metric_value = inter / union
        elif match_metric == "IOS":
            smaller = torch.min(rem_areas, areas[idx])
            match_metric_value = inter / smaller
        else:
            raise ValueError("Unsupported match_metric. Use 'IOU' or 'IOS'.")
  
        # Keep boxes with IoU/IOS less than the threshold
        mask = match_metric_value < match_threshold
        order = order[mask]
    print("Final Bounding Box Count:", len(keep))
    return keep

class OptimizedNMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()

        #print("CLASS TYPE: ", self.class_agnostic)
        print("Original Prediction Count", len(object_predictions_as_torch))
        
        # Store original indices before filtering
        #original_indices = torch.arange(len(object_predictions_as_torch))
        
        # Apply pre-filtering (low confidence and small-area removal)
        filtered_predictions, adaptive_iou = adaptive_filter_predictions(object_predictions_as_torch, conf_threshold=self.conf_threshold, min_area=self.min_area, image_height = self.image_height)
        
        if filtered_predictions.numel() == 0:
            return torch.empty((0, 6))  # Return empty tensor if no detections remain
        else:
            if self.class_agnostic:
                keep = optimized_nms(
                    filtered_predictions, adaptive_iou, match_metric=self.match_metric, conf_threshold=self.conf_threshold, min_area=self.min_area
                )
            else:
                keep = batched_optimized_nms(
                    filtered_predictions, adaptive_iou, match_metric=self.match_metric, conf_threshold=self.conf_threshold, min_area=self.min_area
                )

            selected_object_predictions = object_prediction_list[keep].tolist()
            if not isinstance(selected_object_predictions, list):
                selected_object_predictions = [selected_object_predictions]
    
            return selected_object_predictions



class OptimizedNMSPostprocess_bk(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        #print("Prediction Count Original: ", len(object_predictions_as_torch))
        
        if self.class_agnostic:
            keep = optimized_nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, 	match_metric=self.match_metric, conf_threshold=self.conf_threshold, min_area=self.min_area
            )
        else:
            keep = batched_optimized_nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric, conf_threshold=self.conf_threshold, min_area=self.min_area
            )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions


#*******************************************************************************************#
class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep = nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric
            )
        else:
            keep = batched_nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric
            )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions


class NMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep_to_merge_list = nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(), object_prediction_list[merge_ind].tolist()
                    )
            selected_object_predictions.append(object_prediction_list[keep_ind].tolist())

        return selected_object_predictions


class GreedyNMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep_to_merge_list = greedy_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_greedy_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(), object_prediction_list[merge_ind].tolist()
                    )
            selected_object_predictions.append(object_prediction_list[keep_ind].tolist())

        return selected_object_predictions


class LSNMSPostprocess(PostprocessPredictions):
    # https://github.com/remydubois/lsnms/blob/10b8165893db5bfea4a7cb23e268a502b35883cf/lsnms/nms.py#L62
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        try:
            from lsnms import nms
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please run "pip install lsnms>0.3.1" to install lsnms first for lsnms utilities.'
            )

        if self.match_metric == "IOS":
            NotImplementedError(f"match_metric={self.match_metric} is not supported for LSNMSPostprocess")

        logger.warning("LSNMSPostprocess is experimental and not recommended to use.")

        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_numpy = object_prediction_list.tonumpy()

        boxes = object_predictions_as_numpy[:, :4]
        scores = object_predictions_as_numpy[:, 4]
        class_ids = object_predictions_as_numpy[:, 5].astype("uint8")

        keep = nms(
            boxes, scores, iou_threshold=self.match_threshold, class_ids=None if self.class_agnostic else class_ids
        )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions
