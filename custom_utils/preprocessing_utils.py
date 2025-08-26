# ------------------------------------------------------------------------------------------------
# Utility Functions for Image Preprocessing and Information Retrival
# ------------------------------------------------------------------------------------------------

import os 
import sys
import json
import cv2
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ------------------------------------------------------------------------------------------------
# Returns the total number of files in a specified folder
# ------------------------------------------------------------------------------------------------
def count_files_in_folder(folder_path):
    try:
        total_images =  sum(1 for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)))
        print(f"Total number of files: {total_images}")
    except FileNotFoundError:
        print("The specified folder does not exist.")


# ------------------------------------------------------------------------------------------------
# Prints the largest and smallest size image in a specified folder
# ------------------------------------------------------------------------------------------------
def display_images_info(folder_path):
    max_image = {"name": None, "size": (0, 0), "area": 0}
    min_image = {"name": None, "size": (float("inf"), float("inf")), "area": float("inf")}

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    size = img.size  # (width, height)
                    area = size[0] * size[1]

                    if area > max_image["area"]:
                        max_image = {"name": file, "size": size, "area": area}

                    if area < min_image["area"]:
                        min_image = {"name": file, "size": size, "area": area}
            except:
                continue  # Skip non-image files

    print(f"Largest Image: {max_image['name']} - {max_image['size'][0]}x{max_image['size'][1]}")
    print(f"Smallest Image: {min_image['name']} - {min_image['size'][0]}x{min_image['size'][1]}")


# ------------------------------------------------------------------------------------------------
# Resizes all the images in a specified folder to the defined max_size (width or height)
# ------------------------------------------------------------------------------------------------
def resize_image(image_path, output_folder, max_size=640):
    image = Image.open(image_path)
    width, height = image.size
    
    # Check if resizing is needed
    if width > max_size or height > max_size:
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        print(f"Resized {os.path.basename(image_path)} to {new_width}x{new_height}")
        
    # Save resized image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    image.save(output_path)

# ------------------------------------------------------------------------------------------------
# Processes all the images in a specified folder and utilizes resize_image() to the defined max_size (width or height)
# ------------------------------------------------------------------------------------------------
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
            image_path = os.path.join(input_folder, filename)
            resize_image(image_path, output_folder)


input_folder = "../YOLOX/datasets/COCO_back/test"  # Change this to the actual path
#output_folder = "../../data/DOTAv1.5/val2017"  # Change this to the actual path

# Example usage of functions
#process_images(input_folder, output_folder)
#count_files_in_folder(output_folder)
display_images_info(input_folder)
