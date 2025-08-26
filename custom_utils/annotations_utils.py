# ------------------------------------------------------------------------------------------------
# Utility Functions for DOTAv1.5 COCO annotations Processing and Information Retrival
# ------------------------------------------------------------------------------------------------

import os 
import sys
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib.pyplot as plt

def load_coco_annotations(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data
    
# ------------------------------------------------------------------------------------------------
# Retrives the annotations information for a specific image_id in provided dataset file
# ------------------------------------------------------------------------------------------------
def get_annotations_by_image_id(coco_json_path, image_id, output_filename=None):
    # Load COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Filter annotations for the given image_id
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # Define output file name if not provided
    if output_filename is None:
        output_filename = f"annotations_image_{image_id}.json"

    # Save the result to a JSON file in the current folder
    with open(output_filename, 'w') as f:
        json.dump(annotations, f, indent=4)

    print(f"Saved annotations for image_id {image_id} to {output_filename}")

    return output_filename  # Return the filename for reference


# ------------------------------------------------------------------------------------------------
# Prints the total number of bounding boxes associated with each image 
# ------------------------------------------------------------------------------------------------
def count_bounding_boxes(coco_json_path):
    # Load COCO JSON file
    coco_data = load_coco_json(coco_json_path)

    # Create a mapping of image_id to file_name
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}

    # Count annotations per image
    bbox_count_per_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        bbox_count_per_image[image_id] = bbox_count_per_image.get(image_id, 0) + 1

    # Print file name and number of bounding boxes
    for image_id, bbox_count in bbox_count_per_image.items():
        file_name = image_id_to_file.get(image_id, "Unknown_File")
        print(f"{file_name}: {bbox_count} bounding boxes")

# ------------------------------------------------------------------------------------------------
# Prints the image with maximum and minimum number of bounding boxes
# ------------------------------------------------------------------------------------------------

def count_minmax_bounding_boxes(coco_json_path):
    # Load COCO JSON file
    coco_data = load_coco_json(coco_json_path)

    # Create a mapping of image_id to file_name
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}

    # Count annotations (bounding boxes) per image
    bbox_count_per_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        bbox_count_per_image[image_id] = bbox_count_per_image.get(image_id, 0) + 1

    # Print file name and number of bounding boxes for each image
    for image_id, bbox_count in bbox_count_per_image.items():
        file_name = image_id_to_file.get(image_id, "Unknown_File")
        #print(f"{file_name}: {bbox_count} bounding boxes")
    
    # Ensure that there is at least one annotated image before determining min/max
    if not bbox_count_per_image:
        print("No bounding boxes found.")
        return

    # Determine the image id with maximum and minimum bounding boxes
    max_image_id = max(bbox_count_per_image, key=bbox_count_per_image.get)
    min_image_id = min(bbox_count_per_image, key=bbox_count_per_image.get)

    # Retrieve file names using the image id mapping
    max_file_name = image_id_to_file.get(max_image_id, "Unknown_File")
    min_file_name = image_id_to_file.get(min_image_id, "Unknown_File")
    
    # Print the analysis results for maximum and minimum bounding boxes
    print(f"Image with maximum bounding boxes: {max_file_name} ({bbox_count_per_image[max_image_id]} bounding boxes)")
    print(f"Image with minimum bounding boxes: {min_file_name} ({bbox_count_per_image[min_image_id]} bounding boxes)")

    # Calculate average bounding boxes per image using all images,
    # even those with zero annotations. For this, we use the total number of annotations
    # and the total number of images.
    total_images = len(coco_data['images'])
    total_boxes = len(coco_data['annotations'])
    average_boxes = math.floor(total_boxes / total_images) if total_images > 0 else 0
    print(f"Average bounding boxes per image: {average_boxes}")

# ------------------------------------------------------------------------------------------------
# Validates the COCO annotations file and visualize the bounding boxes in corresponding image file
# -------------------------------------------------------------------------------

def draw_bounding_boxes(image_path, annotations, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return 0  # Return 0 bounding boxes if image can't be read

    bbox_count = 0
    for ann in annotations:
        x, y, w, h = map(int, ann['bbox'])  # Convert bbox to integers
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bbox_count += 1  # Count number of bounding boxes
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path} | Bounding Boxes: {bbox_count}")

    return bbox_count

def validate_and_visualize(coco_json_path, images_folder, output_folder):
    coco_data = load_coco_json(coco_json_path)

    # Map image_id to image file
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(ann)
        else:
            annotations_by_image[image_id] = [ann]

    total_images_processed = 0
    total_bboxes = 0
    skipped_count = 0
    # Process each image that exists in the images folder
    for image_id, file_name in image_id_to_file.items():
        image_path = os.path.join(images_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        if not os.path.exists(image_path):  
            #print(f"Skipping: {file_name} (File not found)")
            skipped_count+= 1
            continue  # Skip missing images

        if image_id in annotations_by_image:
            bbox_count = draw_bounding_boxes(image_path, annotations_by_image[image_id], output_path)
            total_images_processed += 1
            total_bboxes += bbox_count
        else:
            print(f"Skipping: {file_name} (No annotations found)")
    
    print("**********************************************")
    print(f"\nSummary:\n")
    print("**********************************************")
    print(f"Total Images Processed: {total_images_processed}")
    print(f"Total Skipped Images: {skipped_count}")
    #print(f"Total Bounding Boxes Plotted: {total_bboxes}")


def load_coco_json(path):
    """
    Loads a COCO format JSON file.
    
    Parameters:
    - path (str): Path to the JSON file.
    
    Returns:
    - dict: Parsed JSON content.
    """
    with open(path, 'r') as f:
        return json.load(f)

# ------------------------------------------------------------------------------------------------
# Visualize the bounding box count per image using histogram
# ------------------------------------------------------------------------------------------------

def plot_bbox_histogram(coco_json_path, output_folder="output", output_filename="histogram.png"):
    """
    Generates and saves a histogram showing the distribution of bounding boxes per image
    from a COCO formatted JSON file. The x-axis bins span intervals of 50 units and includes
    additional dataset statistics in the plot.

    Parameters:
    - coco_json_path (str): Path to the COCO JSON annotations file.
    - output_folder (str, optional): Directory where the output image will be saved. Defaults to "output".
    - output_filename (str, optional): File name for the saved histogram image. Defaults to "histogram.png".
    """
    
    # Load COCO JSON file
    coco_data = load_coco_json(coco_json_path)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, output_filename)
    
    # Initialize a count list with a default value of 0 for each image.
    counts = [0] * len(coco_data['images'])
    
    # Create a mapping from image_id to its index in the images list.
    image_id_to_index = {img['id']: idx for idx, img in enumerate(coco_data['images'])}
    
    # Update the count for each image based on the annotations.
    for ann in coco_data['annotations']:
        idx = image_id_to_index.get(ann['image_id'])
        if idx is not None:
            counts[idx] += 1

    # Calculate summary statistics
    total_images = len(counts)
    total_bboxes = sum(counts)
    mean_bboxes = total_bboxes / total_images if total_images > 0 else 0
    median_bboxes = np.median(counts) if total_images > 0 else 0

    # Define bins with 50 interval steps.
    # We ensure that our bins cover the entire range of counts
    max_count = max(counts)
    bins = np.arange(0, max_count + 50, 50)
    
    # Create a figure with a specified size for better readability.
    plt.figure(figsize=(10, 6))
    
    # Create the histogram.
    n, bins, patches = plt.hist(counts, bins=bins, edgecolor='black')
    plt.xlabel("Number of Bounding Boxes per Image")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Bounding Boxes per Image")
    
    # Add gridlines for better visual guidance.
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Prepare additional information to be displayed on the histogram.
    stats_text = (
        f"Total Images: {total_images}\n"
        f"Total BBoxes: {total_bboxes}\n"
        f"Mean: {mean_bboxes:.2f}\n"
        f"Median: {median_bboxes:.2f}"
    )
    
    # Place the stats text box in the top-right corner of the plot.
    plt.text(
        0.95, 0.95, stats_text, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    # Save the histogram in the specified folder.
    plt.savefig(output_image_path)
    print(f"Histogram saved as '{output_image_path}'")
    
    # Display the plot.
    plt.show()


def plot_bbox_histogram_2(coco_json_path, output_folder="output", output_filename="histogram_2.png"):
    # Load COCO JSON file
    coco_data = load_coco_json(coco_json_path)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, output_filename)
    
    # Initialize a count list with a default value of 0 for each image.
    counts = [0] * len(coco_data['images'])
    
    # Create a mapping from image_id to its index in the images list.
    image_id_to_index = {img['id']: idx for idx, img in enumerate(coco_data['images'])}
    
    # Update the count for each image based on the annotations.
    for ann in coco_data['annotations']:
        idx = image_id_to_index.get(ann['image_id'])
        if idx is not None:
            counts[idx] += 1
    
    # Create the histogram.
    plt.hist(counts, bins=range(min(counts), max(counts) + 2), edgecolor='black')
    plt.xlabel("Number of Bounding Boxes per Image")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Bounding Boxes per Image")
    
    # Save the histogram in the specified folder.
    plt.savefig(output_image_path)
    print(f"Histogram saved as '{output_image_path}'")
    
    # Display the plot.
    plt.show()


# ------------------------------------------------------------------------------------------------
# Visualize the bounding box count per area pixel using scatter plot
# ------------------------------------------------------------------------------------------------

def visualize_bbox_scatter(coco_json_path, output_folder="output", output_filename="scatter_plot.png"):
    """
    Visualizes the bounding boxes count per image using a scatter plot. The plot uses the image area
    as the x-axis and the bounding boxes count as the y-axis. The color of each marker represents the
    object density (bbox count per image area), which helps in analyzing the correlation between image
    properties and object density.
    
    Parameters:
    - coco_json_path (str): Path to the COCO JSON annotations file.
    - output_folder (str): Directory where the output plot image will be saved.
    - output_filename (str): File name of the saved plot image.
    """
    
    # Load COCO JSON dataset
    coco_data = load_coco_json(coco_json_path)
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, output_filename)
    
    # Prepare dictionaries and lists to accumulate image properties and bbox counts.
    # Create a mapping of image_id to image info.
    image_data = {img["id"]: img for img in coco_data["images"]}
    
    # Initialize a dictionary for bbox counts with image id as key.
    bbox_count_dict = {img["id"]: 0 for img in coco_data["images"]}
    
    # Count the bounding boxes for each image using the annotations.
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id in bbox_count_dict:
            bbox_count_dict[image_id] += 1
    
    # Lists to store computed properties.
    areas = []
    bbox_counts = []
    densities = []  # Density = bounding boxes per unit area
    
    # Iterate through each image and compute the area and density.
    for image_id, img in image_data.items():
        width = img.get("width", 0)
        height = img.get("height", 0)
        area = width * height
        count = bbox_count_dict.get(image_id, 0)
        density = count / area if area > 0 else 0
        
        areas.append(area)
        bbox_counts.append(count)
        densities.append(density)
    
    # Convert lists to numpy arrays for convenience.
    areas = np.array(areas)
    bbox_counts = np.array(bbox_counts)
    densities = np.array(densities)
    
    # Compute Pearson correlation coefficient between image area and bbox count.
    if areas.size > 1:
        correlation = np.corrcoef(areas, bbox_counts)[0, 1]
    else:
        correlation = 0
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot: x-axis = image area, y-axis = bounding boxes count,
    # with color representing object density.
    scatter = plt.scatter(areas, bbox_counts, c=densities, cmap='viridis', alpha=0.7,
                          edgecolors='w', s=100)
    
    plt.xlabel("Image Area (pixels^2)")
    plt.ylabel("Bounding Boxes Count")
    plt.title("Scatter Plot: Bounding Boxes Count vs Image Area")
    
    # Add a color bar indicating object density.
    cbar = plt.colorbar(scatter)
    cbar.set_label("Object Density (bboxes per pixels^2)")
    
    # Calculate additional statistics for annotation.
    total_images = len(areas)
    total_bboxes = np.sum(bbox_counts)
    mean_bboxes = np.mean(bbox_counts)
    median_bboxes = np.median(bbox_counts)
    
    stats_text = (
        f"Total Images: {total_images}\n"
        f"Total BBoxes: {total_bboxes}\n"
        f"Mean BBoxes: {mean_bboxes:.2f}\n"
        f"Median BBoxes: {median_bboxes:.2f}\n"
        f"Pearson r (Area vs Count): {correlation:.2f}"
    )
    
    # Place the statistics box in the upper right corner.
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add gridlines for better readability.
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the scatter plot.
    plt.savefig(output_image_path, bbox_inches='tight')
    print(f"Scatter plot saved as '{output_image_path}'")
    
    # Display the plot.
    plt.show()


# ------------------------------------------------------------------------------------------------
# Prints the total number of bounding boxes associated with specfic category id 
# ------------------------------------------------------------------------------------------------
def count_annotations_by_category(coco_json_path):
    # Load the COCO JSON file using your existing helper function.
    coco_data = load_coco_json(coco_json_path)
    
    # Initialize a dictionary to store counts for each category_id.
    category_annotation_count = {}
    
    # Loop through all annotations in the JSON.
    for ann in coco_data.get("annotations", []):
        # Retrieve the category id from the annotation.
        category_id = ann.get("category_id")
        # Increment the count for the category_id.
        if category_id is not None:
            category_annotation_count[category_id] = category_annotation_count.get(category_id, 0) + 1
    
    # Print the total annotations count for each category.
    for category_id, count in category_annotation_count.items():
        print(f"Category ID {category_id}: {count} annotations")
    
    # Return the counts dictionary.
    return category_annotation_count

# ------------------------------------------------------------------------------------------------
# Prints the total number of bounding boxes associated with each image 
# ------------------------------------------------------------------------------------------------
def count_annotations_for_category(coco_json_path, target_category_id):
    # Load the COCO JSON file using your existing helper function.
    coco_data = load_coco_json(coco_json_path)
    
    # Initialize a count for the specified category
    count = 0
    
    # Loop through all annotations in the JSON.
    for ann in coco_data.get("annotations", []):
        # Check if this annotation matches the specified category_id.
        if ann.get("category_id") == target_category_id:
            count += 1
            
    # Print and return the count of annotations for the specified category.
    print(f"Category ID {target_category_id}: {count} annotations")
    return count


# ------------------------------------------------------------------------------------------------
# Retrieve image ID, width, and height for a given file name from a COCO JSON file
# ------------------------------------------------------------------------------------------------
def get_image_info_from_filename(coco_json_path, file_name):
    """
    Retrieve image ID, width, and height for a given file name from a COCO JSON file.

    Args:
        coco_json_path (str): Path to the COCO JSON file.
        file_name (str): Name of the image file.

    Returns:
        dict or None: Dictionary containing 'id', 'width', and 'height', or None if not found.
    """
    # Load COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Search for the image in the dataset
    image_info = next((img for img in coco_data['images'] if img['file_name'] == file_name), None)

    if image_info:
        print("************Image Info*******************")
        print(image_info)
        return {
            "id": image_info['id'],
            "width": image_info['width'],
            "height": image_info['height']
        }
    else:
        print(f"Error: Image file '{file_name}' not found in COCO annotations.")
        return None

# Example usage
image_folder = "../YOLOX/datasets/COCO_back/test"  # Change this to the actual path
annotation_file = "../../YOLOX/datasets/COCO_back/annotations/instances_dev2017.json"  # Change this to the actual path
annotation_file1 = "../../data/VisDrone2COCO/COCO/annotations/visdrone_coco_test.json"
ann2 = '../subset_visdrone_test_data_15_v5.json'
ann = '../../sahi/subset_vis_test_data_15.json'
output_folder = "../YOLOX/datasets/COCO_back/results"  # Change this to the actual path
image_id = 126  # Replace with desired image_id
file_name = "9999979_00000_d_0000002.jpg"  # Replace with your image filename
category_id = 1265
output_folder = "./output/"
#validate_and_visualize(annotation_file, image_folder, output_folder)
count_bounding_boxes(ann2)
#count_minmax_bounding_boxes(annotation_file)
#get_annotations_by_image_id(ann, image_id)
# Example: Count annotations for category_id = 3 (e.g., "car")

#count_annotations_by_category(annotation_file1)
#total_annotations = count_annotations_for_category(annotation_file1, category_id)
#plot_bbox_histogram(annotation_file1, output_folder, "plot_hist.png")
#visualize_bbox_scatter(annotation_file1, output_folder)
#print(f"Total number of annotations for category_id {category_id}: {total_annotations}")
#get_image_info_from_filename(annotation_file1, file_name)