# ------------------------------------------------------------------------------------------------
# Utility Functions for DOTAV1.5 COCO annotations Visualization
# ------------------------------------------------------------------------------------------------

import json
import os
import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

def load_coco_annotations(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

# ------------------------------------------------------------------------------------------------
# Provides the number of object instances in each categories
# ------------------------------------------------------------------------------------------------
def plot_class_distribution(coco_data, save_folder="output_plots"):
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Extract category names and counts
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_counts = Counter([ann['category_id'] for ann in coco_data['annotations']])
    
    class_names = [category_id_to_name[cat_id] for cat_id in category_counts.keys()]
    counts = list(category_counts.values())

    # Create the plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x=class_names, y=counts, color="lightblue")

    # Add text annotations (instance count) above each bar
    for i, count in enumerate(counts):
        ax.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')  # Adding 5 for spacing above the bar

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Categories")
    plt.ylabel("Number of Instances")
    plt.title("Class Distribution per category in DOTAv1.5 Training Dataset")

    # Save the plot to the specified folder
    plot_filename = os.path.join(save_folder, "dota_class_distribution_train.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()  # Close the plot to avoid display

    print(f"Plot saved to {plot_filename}")


# ------------------------------------------------------------------------------------------------
# Displays the heatmaps of the objects in an image file
# ------------------------------------------------------------------------------------------------
def plot_heatmap(image_shape, annotations, save_folder="output_heatmaps"):
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Initialize the heatmap (zeros)
    heatmap = np.zeros(image_shape[:2])

    # Iterate through each annotation to populate the heatmap
    for ann in annotations:
        x, y, w, h = map(int, ann['bbox'])
        heatmap[y:y+h, x:x+w] += 1  # Increment the corresponding area in the heatmap

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap, cmap="hot", cbar=True)
    plt.title("Object Density Heatmap")

    # Save the heatmap to the specified folder
    plot_filename = os.path.join(save_folder, f"heatmap_image_{image_shape[0]}x{image_shape[1]}.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()  # Close the plot to avoid display

    print(f"Heatmap saved to {plot_filename}")



# Example usage
image_id = 1  # Replace with valid image_id
coco_json_path = "../YOLOX/datasets/DOTAv1.5/annotations/DOTA_1.5_train.json"  # Replace with actual path
coco_data = load_coco_annotations(coco_json_path)
image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
image_shape = (image_info['height'], image_info['width'], 3)


# Example usage
#plot_class_distribution(coco_data, save_folder="plots")
#plot_heatmap(image_shape, annotations, save_folder="plots")
plot_class_distribution(coco_data, save_folder="plots")