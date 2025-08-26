import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_coco_bboxes(coco_json_path, images_folder):
    # Load COCO annotations
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Create a dictionary mapping image_id to image metadata
    image_info = {img["id"]: img for img in coco_data["images"]}

    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]  # [x, y, width, height]

        if image_id not in image_info:
            print(f"Warning: Image ID {image_id} not found in images metadata. Skipping...")
            continue

        image_filename = image_info[image_id]["file_name"]
        image_path = os.path.join(images_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_filename} not found. Skipping...")
            continue

        # Open image
        image = Image.open(image_path)
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Draw bounding box
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Show image with bounding box
        plt.title(f"Image: {image_filename} - Bounding Box: {bbox}")
        plt.axis("off")
        plt.show()

# Example usage
coco_json_path = "C:/path/to/your/annotations.json"  # Path to COCO JSON file
images_folder = "C:/path/to/your/images"  # Path to images folder
draw_coco_bboxes(coco_json_path, images_folder)
