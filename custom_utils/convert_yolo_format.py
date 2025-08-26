import os
import sys

def convert_quad_to_yolo(input_folder: str, output_folder: str):
    """
    Converts VisDrone-style quadrilateral annotations (class x1 y1 x2 y2 x3 y3 x4 y4)
    into standard YOLO format (class x_center y_center width height), saving results
    in the specified output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.txt'):
            continue
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                parts = line.strip().split()
                if len(parts) < 9:
                    # Skip malformed lines
                    continue
                
                class_id = parts[0]
                coords = list(map(float, parts[1:9]))
                xs = coords[0::2]  #start from index 0 element with 2 step each [x1,x2,x3,x4]
                ys = coords[1::2] #start from index 1 element with 2 step each [y1, y2, y3, y4]
                
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                
                # Write in YOLO normalized format
                outfile.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_quad_to_yolo.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    convert_quad_to_yolo(input_folder, output_folder)

# Example usage:
# !python convert_yolo_format.py runs/slice_coco/subset_train_images_128_05_550/labels runs/slice_coco/subset_train_images_128_05_550/yolo_labels
