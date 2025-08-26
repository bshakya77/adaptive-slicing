# ------------------------------------------------------------------------------------------------
# Utility Functions to update the orginal COCO annotations file as per resized image size 
# ------------------------------------------------------------------------------------------------

import os
import json
from PIL import Image

def update_coco_annotations(coco_json_path, images_folder, output_json_path="../../data/DOTAv1.5/annotations/DOTA_1.5_val.json"):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Store image info in a dictionary (image_id -> image metadata)
    image_info = {img["file_name"]: img for img in coco_data["images"]}

    for img in coco_data["images"]:
        image_filename = img["file_name"]
        image_path = os.path.join(images_folder, image_filename)

        # Check if the image exists in the resized folder
        if not os.path.exists(image_path):
            print(f"Warning: Resized image {image_filename} not found. Skipping...")
            continue

        # Get new image dimensions
        with Image.open(image_path) as image:
            new_width, new_height = image.size

        # Get original dimensions
        old_width, old_height = img["width"], img["height"]

        # Scale factor
        scale_x = new_width / old_width
        scale_y = new_height / old_height

        print("Scale X:", scale_x)
        print("Scale Y:", scale_y)
        
        # Update image metadata
        img["width"] = new_width
        img["height"] = new_height

        # Update corresponding annotations
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == img["id"]:
                x, y, w, h = annotation["bbox"]
                annotation["bbox"] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]

    # Save the updated COCO JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Updated annotations saved to {output_json_path}")

# Example usage
coco_json_path = "../../YOLOX/datasets/DOTAv1.5/annotations/DOTA_1.5_val.json"  # Path to COCO JSON file
images_folder = "../../data/DOTAv1.5/val2017"  # Path to resized images folder
update_coco_annotations(coco_json_path, images_folder)
