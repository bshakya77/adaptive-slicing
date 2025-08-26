import json
import numpy as np
import pandas as pd
import os

def load_coco_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def categorize_images_to_excel(coco_json_path, output_folder="output", excel_filename="categorized_images.xlsx"):
    coco_data = load_coco_json(coco_json_path)
    images = coco_data['images']
    annotations = coco_data['annotations']

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Count bounding boxes per image
    bbox_counts = {img['id']: 0 for img in images}
    for ann in annotations:
        img_id = ann['image_id']
        bbox_counts[img_id] += 1

    counts = np.array(list(bbox_counts.values()))

    # Compute percentile thresholds
    p20, p40, p60, p80 = np.percentile(counts, [20, 40, 60, 80])

    print("Computed thresholds based on percentiles:")
    print(f"  Sparse: bbox count < {p20:.1f}")
    print(f"  Low: {p20:.1f} ≤ bbox count < {p40:.1f}")
    print(f"  Medium: {p40:.1f} ≤ bbox count < {p60:.1f}")
    print(f"  High: {p60:.1f} ≤ bbox count < {p80:.1f}")
    print(f"  Dense: bbox count ≥ {p80:.1f}\n")

    # Categorize images
    categories = {'Sparse': [], 'Low': [], 'Medium': [], 'High': [], 'Dense': []}

    for img in images:
        count = bbox_counts[img['id']]
        filename_only = os.path.basename(img.get('file_name', ''))
        width = img.get('width', 0)
        height = img.get('height', 0)
        area = width * height
        img_info = {
            'id': img['id'],
            'file_name': filename_only,
            'width': width,
            'height': height,
            'area': area,
            'bbox_count': count
        }

        if count < p20:
            categories['Sparse'].append(img_info)
        elif count < p40:
            categories['Low'].append(img_info)
        elif count < p60:
            categories['Medium'].append(img_info)
        elif count < p80:
            categories['High'].append(img_info)
        else:
            categories['Dense'].append(img_info)

    # Save categorized image info to Excel
    excel_path = os.path.join(output_folder, excel_filename)
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        for cat, imgs in categories.items():
            df = pd.DataFrame(imgs).sort_values(by='bbox_count')
            df.to_excel(writer, sheet_name=cat, index=False)

    print(f"Categorized images information saved to: '{excel_path}'")

# Usage example:
data_path = '../../data/VisDrone2COCO/COCO/annotations/visdrone_coco_test.json'
categorize_images_to_excel(data_path, output_folder="output_excel")