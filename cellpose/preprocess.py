import cv2
import numpy as np
import os

def apply_clahe(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    clahe_img = clahe.apply(img)

    # Save output
    cv2.imwrite(output_path, clahe_img)
    print(f"Saved CLAHE corrected image to {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(script_dir, "stemcell-dataset")
    output_root = os.path.join(script_dir, "processed-dataset")

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    for root, dirs, files in os.walk(input_root):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                input_path = os.path.join(root, file)
                
                # Determine relative path from input root
                rel_path = os.path.relpath(root, input_root)
                # Construct output directory path
                output_dir = os.path.join(output_root, rel_path)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_path = os.path.join(output_dir, file)
                
                print(f"Processing {file}...")
                apply_clahe(input_path, output_path)
