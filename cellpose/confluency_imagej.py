"""
Cell Confluency Calculator using ImageJ-style Processing

This script measures cell confluency using traditional image processing methods
similar to ImageJ's approach:
1. CLAHE preprocessing for contrast enhancement
2. Gaussian blur for noise reduction
3. Otsu's automatic thresholding (ImageJ's default auto-threshold)
4. Morphological operations (opening/closing) to clean up the mask
5. Confluency calculation as percentage of cell area

This approach is faster than deep learning methods but may be less accurate
for complex cell morphologies.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def apply_clahe(img_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    This is equivalent to ImageJ's "Enhance Local Contrast (CLAHE)" plugin.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)


def apply_gaussian_blur(img, kernel_size=5):
    """
    Apply Gaussian blur for noise reduction.
    Equivalent to ImageJ's Process > Filters > Gaussian Blur.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def otsu_threshold(img):
    """
    Apply Otsu's automatic thresholding.
    This is equivalent to ImageJ's Image > Adjust > Threshold > Auto (Otsu).
    
    Returns:
        binary_mask: Binary mask where cells are white (255) and background is black (0)
        threshold_value: The automatically determined threshold value
    """
    threshold_value, binary_mask = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary_mask, threshold_value


def apply_morphological_operations(binary_mask, kernel_size=3, iterations=1):
    """
    Apply morphological operations to clean up the binary mask.
    Equivalent to ImageJ's Process > Binary > Open/Close.
    
    - Opening: Removes small noise (erode then dilate)
    - Closing: Fills small holes (dilate then erode)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening to remove small noise
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Closing to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return closed


def calculate_confluency_imagej(
    image_path: str,
    clip_limit: float = 2.0,
    blur_kernel: int = 5,
    morph_kernel: int = 3,
    morph_iterations: int = 1,
    invert: bool = False,
    save_outputs: bool = True,
    output_dir: str = None
) -> dict:
    """
    Calculate cell confluency using ImageJ-style image processing.
    
    Args:
        image_path: Path to the input image
        clip_limit: CLAHE clip limit (default 2.0)
        blur_kernel: Gaussian blur kernel size (default 5)
        morph_kernel: Morphological operation kernel size (default 3)
        morph_iterations: Number of morphological iterations (default 1)
        invert: If True, invert the threshold (for dark cells on light background)
        save_outputs: Whether to save output images
        output_dir: Directory to save outputs (defaults to image directory)
    
    Returns:
        dict containing confluency results and intermediate outputs
    """
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    if output_dir == '':
        output_dir = '.'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # =========================================================================
    # STEP 1: Load original image
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: Loading Original Image")
    print(f"{'='*60}")
    
    # Load image
    img_original = cv2.imread(image_path)
    if img_original is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    height, width = img_gray.shape
    total_pixels = height * width
    
    print(f"Loaded image: {image_path}")
    print(f"Image size: {width}x{height}")
    
    # =========================================================================
    # STEP 2: Preprocessing (CLAHE + Gaussian Blur)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Preprocessing (CLAHE + Blur)")
    print(f"{'='*60}")
    
    # Apply CLAHE
    print(f"Applying CLAHE (clip_limit={clip_limit})...")
    img_clahe = apply_clahe(img_gray, clip_limit=clip_limit)
    
    # Apply Gaussian blur
    print(f"Applying Gaussian blur (kernel={blur_kernel})...")
    img_blurred = apply_gaussian_blur(img_clahe, kernel_size=blur_kernel)
    img_blurred = img_clahe(img_clahe, kernel_size=blur_kernel)

    
    print("Preprocessing complete")
    
    # =========================================================================
    # STEP 3: Otsu's Thresholding (ImageJ Auto Threshold)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: Otsu's Automatic Thresholding")
    print(f"{'='*60}")
    
    # Apply Otsu's threshold
    binary_mask, threshold_value = otsu_threshold(img_blurred)
    
    # Invert if needed (for dark cells on light background)
    if invert:
        binary_mask = cv2.bitwise_not(binary_mask)
        print(f"Threshold inverted (dark objects on light background)")
    
    print(f"Otsu threshold value: {threshold_value:.1f}")
    
    # Apply morphological operations
    print(f"Applying morphological operations (kernel={morph_kernel}, iterations={morph_iterations})...")
    binary_mask_cleaned = apply_morphological_operations(
        binary_mask, 
        kernel_size=morph_kernel, 
        iterations=morph_iterations
    )
    
    # Calculate confluency
    cell_area_pixels = np.sum(binary_mask_cleaned > 0)
    confluency = (cell_area_pixels / total_pixels) * 100
    
    print(f"Cell pixels: {cell_area_pixels:,} / {total_pixels:,}")
    print(f"Confluency: {confluency:.2f}%")
    
    # =========================================================================
    # FINAL: Create side-by-side comparison
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL: Creating Process Visualization")
    print(f"{'='*60}")
    
    if save_outputs:
        final_path = os.path.join(output_dir, f"{base_name}_imagej_overview.png")
        
        # Create 1x3 horizontal layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Step 1: Original Image
        ax1 = axes[0]
        ax1.imshow(img_rgb)
        ax1.set_title('Step 1: Original Image', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Step 2: CLAHE + Blur (preprocessed)
        ax2 = axes[1]
        ax2.imshow(img_blurred, cmap='gray')
        ax2.set_title('Step 2: Preprocessed\n(CLAHE + Gaussian Blur)', 
                      fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        # Step 3: Binary Mask with Confluency
        ax3 = axes[2]
        cmap_binary = LinearSegmentedColormap.from_list('binary_cells', ['black', '#00FF00'])
        ax3.imshow(binary_mask_cleaned, cmap=cmap_binary)
        ax3.set_title(f'Step 3: Binary Mask (Otsu)\nConfluency: {confluency:.1f}%', 
                      fontsize=14, fontweight='bold', pad=10)
        ax3.axis('off')
        
        # Add overall title
        fig.suptitle(f'Cell Confluency Analysis (ImageJ-style)\n'
                     f'Otsu Threshold: {threshold_value:.0f} | '
                     f'CLAHE: {clip_limit} | Blur: {blur_kernel}',
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(final_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {final_path}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("CONFLUENCY ANALYSIS COMPLETE (ImageJ-style)")
    print(f"{'='*60}")
    print(f"Otsu threshold: {threshold_value:.1f}")
    print(f"Total image area: {total_pixels:,} pixels")
    print(f"Cell area: {cell_area_pixels:,} pixels")
    print(f"Cell confluency: {confluency:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'confluency': confluency,
        'cell_area_pixels': int(cell_area_pixels),
        'total_area_pixels': int(total_pixels),
        'threshold_value': float(threshold_value),
        'binary_mask': binary_mask_cleaned,
        'preprocessed_image': img_blurred
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate cell confluency using ImageJ-style processing'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image'
    )
    parser.add_argument(
        '--clip_limit', '-c',
        type=float,
        default=2.0,
        help='CLAHE clip limit (default: 2.0)'
    )
    parser.add_argument(
        '--blur', '-b',
        type=int,
        default=5,
        help='Gaussian blur kernel size (default: 5)'
    )
    parser.add_argument(
        '--morph_kernel', '-mk',
        type=int,
        default=3,
        help='Morphological kernel size (default: 3)'
    )
    parser.add_argument(
        '--morph_iter', '-mi',
        type=int,
        default=1,
        help='Morphological iterations (default: 1)'
    )
    parser.add_argument(
        '--invert', '-i',
        action='store_true',
        help='Invert threshold (for dark cells on light background)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output files'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Run confluency calculation
    results = calculate_confluency_imagej(
        image_path=args.image_path,
        clip_limit=args.clip_limit,
        blur_kernel=args.blur,
        morph_kernel=args.morph_kernel,
        morph_iterations=args.morph_iter,
        invert=args.invert,
        save_outputs=not args.no_save,
        output_dir=args.output_dir
    )
    
    print(f"\n{'#'*60}")
    print(f"FINAL CONFLUENCY: {results['confluency']:.2f}%")
    print(f"{'#'*60}")
