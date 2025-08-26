import os
import glob
import random
import shutil

def create_subset_images(input_folder, output_folder, subset_count):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Supported image extensions (you can add more if needed)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.txt']
    
    # Get list of all image files in the input folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # Check if there are images available
    if not image_files:
        print("No image files found in the input folder.")
        return
    
    # If subset_count is greater than available images, use all images
    if subset_count > len(image_files):
        print(f"Requested subset count {subset_count} is greater than available images ({len(image_files)}). Using all images.")
        subset_count = len(image_files)
    
    # Randomly select a subset of images
    subset_images = random.sample(image_files, subset_count)
    
    # Copy each selected image to the output folder
    for image_path in subset_images:
        filename = os.path.basename(image_path)
        destination_path = os.path.join(output_folder, filename)
        shutil.copy(image_path, destination_path)
        print(f"Copied {filename} to {output_folder}")

    print("Subset creation completed.")

# Example usage:
input_folder = "./runs/slice_coco/train2017_128_05/images"  # Replace with your input folder path
output_folder = "./runs/slice_coco/subset_train_images_128_05_1500/images"  # Replace with your output folder path
subset_count = 1500  # Change to the desired number of images in the subset

create_subset_images(input_folder, output_folder, subset_count)
