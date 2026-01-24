"""
Cell Confluency Calculator using CellProfiler-style Processing

This script measures cell confluency using image processing methods similar to
CellProfiler's algorithms:
1. CLAHE preprocessing for contrast enhancement  
2. Gaussian/Median smoothing for noise reduction
3. Various thresholding methods (Otsu, Triangle, etc.)
4. Morphological operations and object identification
5. Confluency calculation

Uses only OpenCV, numpy, scipy, and matplotlib - no additional dependencies required.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from scipy import ndimage


def apply_clahe(img_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Similar to CellProfiler's CorrectIlluminationCalculate module.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)


def apply_gaussian_smoothing(img, sigma=2):
    """
    Apply Gaussian smoothing (CellProfiler's Smooth module with Gaussian method).
    Uses OpenCV's GaussianBlur.
    """
    # Convert sigma to kernel size (must be odd)
    ksize = int(sigma * 4) | 1  # Ensure odd
    if ksize < 3:
        ksize = 3
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_median_filter(img, size=3):
    """
    Apply median filter (CellProfiler's Smooth module with Median method).
    """
    # Ensure size is odd
    if size % 2 == 0:
        size += 1
    return cv2.medianBlur(img, size)


def threshold_otsu_cv(img):
    """
    Otsu's thresholding using OpenCV.
    Returns threshold value (normalized to 0-1 if input is 0-1).
    """
    # If image is float, convert to uint8 for OpenCV
    if img.dtype == np.float64 or img.dtype == np.float32:
        img_uint8 = (img * 255).astype(np.uint8)
        thresh_val, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_val / 255.0
    else:
        thresh_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_val


def threshold_triangle_cv(img):
    """
    Triangle thresholding using OpenCV.
    """
    if img.dtype == np.float64 or img.dtype == np.float32:
        img_uint8 = (img * 255).astype(np.uint8)
        thresh_val, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return thresh_val / 255.0
    else:
        thresh_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return thresh_val


def threshold_li(img):
    """
    Li's Minimum Cross Entropy thresholding.
    Implementation of Li's iterative method.
    """
    # Ensure image is in 0-1 range
    if img.max() > 1:
        img = img / 255.0
    
    # Initial threshold estimate
    threshold = np.mean(img)
    
    # Iteratively refine
    for _ in range(100):
        # Split pixels by threshold
        foreground = img[img > threshold]
        background = img[img <= threshold]
        
        if len(foreground) == 0 or len(background) == 0:
            break
        
        # Calculate mean of each region
        mean_fg = np.mean(foreground)
        mean_bg = np.mean(background)
        
        if mean_fg <= 0 or mean_bg <= 0:
            break
        
        # New threshold (cross-entropy minimum)
        new_threshold = (mean_fg - mean_bg) / (np.log(mean_fg) - np.log(mean_bg))
        
        # Check convergence
        if abs(new_threshold - threshold) < 1e-6:
            break
        
        threshold = new_threshold
    
    return threshold


def threshold_yen(img):
    """
    Yen's thresholding method.
    Maximizes the correlation between the distribution before and after thresholding.
    """
    if img.max() > 1:
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    else:
        img_uint8 = (img * 255).astype(np.uint8)
    
    # Calculate histogram
    hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    
    # Normalize histogram
    hist = hist / hist.sum()
    
    # Cumulative sums
    cum_sum = np.cumsum(hist)
    cum_sum_sq = np.cumsum(hist ** 2)
    
    # Calculate Yen's criterion for each threshold
    yen_criterion = np.zeros(256)
    for t in range(256):
        if cum_sum[t] > 0 and (1 - cum_sum[t]) > 0:
            # First class entropy
            p1 = cum_sum[t]
            s1 = cum_sum_sq[t]
            if p1 > 0:
                yen1 = -np.log(s1 / (p1 * p1) + 1e-10)
            else:
                yen1 = 0
            
            # Second class entropy
            p2 = 1 - cum_sum[t]
            s2 = cum_sum_sq[-1] - cum_sum_sq[t]
            if p2 > 0:
                yen2 = -np.log(s2 / (p2 * p2) + 1e-10)
            else:
                yen2 = 0
            
            yen_criterion[t] = yen1 + yen2
    
    # Find threshold that maximizes criterion
    threshold = np.argmax(yen_criterion)
    
    return threshold / 255.0 if img.max() <= 1 else threshold


def threshold_robust_background(img, lower_fraction=0.05, upper_fraction=0.05, deviations=2):
    """
    RobustBackground thresholding (CellProfiler's method).
    
    Calculates threshold based on the mean and standard deviation of the
    background pixels (excluding outliers).
    """
    # Flatten and sort
    pixels = img.flatten()
    pixels_sorted = np.sort(pixels)
    
    # Remove outliers
    n = len(pixels_sorted)
    lower_idx = int(n * lower_fraction)
    upper_idx = int(n * (1 - upper_fraction))
    
    # Calculate stats on the remaining pixels
    background_pixels = pixels_sorted[lower_idx:upper_idx]
    mean_bg = np.mean(background_pixels)
    std_bg = np.std(background_pixels)
    
    # Threshold is mean + deviations * std
    threshold = mean_bg + deviations * std_bg
    
    return threshold


def identify_primary_objects(binary_mask, min_size=15, max_size=None, fill_holes=True):
    """
    Identify primary objects (similar to CellProfiler's IdentifyPrimaryObjects).
    
    This performs:
    1. Connected component labeling
    2. Size filtering
    3. Hole filling
    """
    # Fill holes if requested
    if fill_holes:
        binary_mask = ndimage.binary_fill_holes(binary_mask)
    
    # Label connected components
    labeled, num_features = ndimage.label(binary_mask)
    
    # Size filtering
    if min_size > 0 or max_size is not None:
        sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
        
        # Create mask for valid sizes
        for i, size in enumerate(sizes, start=1):
            if size < min_size:
                labeled[labeled == i] = 0
            elif max_size is not None and size > max_size:
                labeled[labeled == i] = 0
        
        # Relabel
        labeled, num_features = ndimage.label(labeled > 0)
    
    return labeled, num_features


def calculate_confluency_cellprofiler(
    image_path: str,
    smoothing_method: str = 'gaussian',
    smoothing_size: float = 2.0,
    threshold_method: str = 'otsu',
    clip_limit: float = 2.0,
    min_object_size: int = 15,
    fill_holes: bool = True,
    apply_clahe_preprocessing: bool = True,
    save_outputs: bool = True,
    output_dir: str = None
) -> dict:
    """
    Calculate cell confluency using CellProfiler-style image processing.
    
    Args:
        image_path: Path to the input image
        smoothing_method: 'gaussian', 'median', or 'none'
        smoothing_size: Size parameter for smoothing
        threshold_method: 'otsu', 'li', 'yen', 'triangle', or 'robust_background'
        clip_limit: CLAHE clip limit (default 2.0)
        min_object_size: Minimum object size in pixels
        fill_holes: Whether to fill holes in detected objects
        apply_clahe_preprocessing: Whether to apply CLAHE before thresholding
        save_outputs: Whether to save output images
        output_dir: Directory to save outputs
    
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
    # STEP 2: Preprocessing (CellProfiler-style)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Preprocessing (CellProfiler-style)")
    print(f"{'='*60}")
    
    img_processed = img_gray.copy()
    
    # Apply CLAHE
    if apply_clahe_preprocessing:
        print(f"Applying CLAHE (clip_limit={clip_limit})...")
        img_processed = apply_clahe(img_processed, clip_limit=clip_limit)
    
    # Apply smoothing
    if smoothing_method == 'gaussian':
        print(f"Applying Gaussian smoothing (sigma={smoothing_size})...")
        img_processed = apply_gaussian_smoothing(img_processed, sigma=smoothing_size)
    elif smoothing_method == 'median':
        print(f"Applying Median filter (size={int(smoothing_size)})...")
        img_processed = apply_median_filter(img_processed, size=int(smoothing_size))
    elif smoothing_method != 'none':
        print(f"Unknown smoothing method: {smoothing_method}, skipping...")
    
    # Normalize to 0-1 for thresholding
    img_normalized = img_processed.astype(np.float64) / 255.0
    
    print("Preprocessing complete")
    
    # =========================================================================
    # STEP 3: Thresholding (CellProfiler IdentifyPrimaryObjects)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 3: Thresholding ({threshold_method})")
    print(f"{'='*60}")
    
    # Calculate threshold based on method
    if threshold_method == 'otsu':
        threshold_value = threshold_otsu_cv(img_normalized)
        print(f"Otsu threshold: {threshold_value:.4f}")
    elif threshold_method == 'li':
        threshold_value = threshold_li(img_normalized)
        print(f"Li (Minimum Cross Entropy) threshold: {threshold_value:.4f}")
    elif threshold_method == 'yen':
        threshold_value = threshold_yen(img_normalized)
        print(f"Yen threshold: {threshold_value:.4f}")
    elif threshold_method == 'triangle':
        threshold_value = threshold_triangle_cv(img_normalized)
        print(f"Triangle threshold: {threshold_value:.4f}")
    elif threshold_method == 'robust_background':
        threshold_value = threshold_robust_background(img_normalized)
        print(f"RobustBackground threshold: {threshold_value:.4f}")
    else:
        # Default to Otsu
        threshold_value = threshold_otsu_cv(img_normalized)
        print(f"Unknown method '{threshold_method}', using Otsu: {threshold_value:.4f}")
    
    # Apply threshold
    binary_mask = img_normalized > threshold_value
    
    # Identify objects (with size filtering and hole filling)
    print(f"Identifying objects (min_size={min_object_size}, fill_holes={fill_holes})...")
    labeled_objects, num_objects = identify_primary_objects(
        binary_mask,
        min_size=min_object_size,
        fill_holes=fill_holes
    )
    
    # Create final binary mask from labeled objects
    final_binary_mask = labeled_objects > 0
    
    # Calculate confluency
    cell_area_pixels = np.sum(final_binary_mask)
    confluency = (cell_area_pixels / total_pixels) * 100
    
    print(f"Objects detected: {num_objects}")
    print(f"Cell pixels: {cell_area_pixels:,} / {total_pixels:,}")
    print(f"Confluency: {confluency:.2f}%")
    
    # =========================================================================
    # FINAL: Create side-by-side comparison
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL: Creating Process Visualization")
    print(f"{'='*60}")
    
    if save_outputs:
        final_path = os.path.join(output_dir, f"{base_name}_cellprofiler_overview.png")
        
        # Create 1x3 horizontal layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Step 1: Original Image
        ax1 = axes[0]
        ax1.imshow(img_rgb)
        ax1.set_title('Step 1: Original Image', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Step 2: Preprocessed
        ax2 = axes[1]
        ax2.imshow(img_processed, cmap='gray')
        preprocess_title = 'Step 2: Preprocessed\n'
        if apply_clahe_preprocessing:
            preprocess_title += f'(CLAHE + {smoothing_method.capitalize()})'
        else:
            preprocess_title += f'({smoothing_method.capitalize()})'
        ax2.set_title(preprocess_title, fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        
        # Step 3: Binary Mask with Confluency
        ax3 = axes[2]
        cmap_binary = LinearSegmentedColormap.from_list('binary_cells', ['black', '#00FF00'])
        ax3.imshow(final_binary_mask, cmap=cmap_binary)
        ax3.set_title(f'Step 3: Binary Mask ({threshold_method})\n'
                      f'Confluency: {confluency:.1f}% | Objects: {num_objects}', 
                      fontsize=14, fontweight='bold', pad=10)
        ax3.axis('off')
        
        # Add overall title
        fig.suptitle(f'Cell Confluency Analysis (CellProfiler-style)\n'
                     f'Threshold: {threshold_method} ({threshold_value:.3f}) | '
                     f'Smoothing: {smoothing_method}',
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(final_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {final_path}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("CONFLUENCY ANALYSIS COMPLETE (CellProfiler-style)")
    print(f"{'='*60}")
    print(f"Threshold method: {threshold_method}")
    print(f"Threshold value: {threshold_value:.4f}")
    print(f"Objects detected: {num_objects}")
    print(f"Total image area: {total_pixels:,} pixels")
    print(f"Cell area: {cell_area_pixels:,} pixels")
    print(f"Cell confluency: {confluency:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'confluency': confluency,
        'cell_area_pixels': int(cell_area_pixels),
        'total_area_pixels': int(total_pixels),
        'threshold_method': threshold_method,
        'threshold_value': float(threshold_value),
        'num_objects': num_objects,
        'binary_mask': final_binary_mask,
        'labeled_objects': labeled_objects,
        'preprocessed_image': img_processed
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate cell confluency using CellProfiler-style processing'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=str,
        default='otsu',
        choices=['otsu', 'li', 'yen', 'triangle', 'robust_background'],
        help='Thresholding method (default: otsu)'
    )
    parser.add_argument(
        '--smoothing', '-s',
        type=str,
        default='gaussian',
        choices=['gaussian', 'median', 'none'],
        help='Smoothing method (default: gaussian)'
    )
    parser.add_argument(
        '--smoothing_size', '-ss',
        type=float,
        default=2.0,
        help='Smoothing size parameter (default: 2.0)'
    )
    parser.add_argument(
        '--clip_limit', '-c',
        type=float,
        default=2.0,
        help='CLAHE clip limit (default: 2.0)'
    )
    parser.add_argument(
        '--min_size', '-ms',
        type=int,
        default=15,
        help='Minimum object size in pixels (default: 15)'
    )
    parser.add_argument(
        '--no-clahe',
        action='store_true',
        help='Disable CLAHE preprocessing'
    )
    parser.add_argument(
        '--no-fill-holes',
        action='store_true',
        help='Disable hole filling'
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
    results = calculate_confluency_cellprofiler(
        image_path=args.image_path,
        smoothing_method=args.smoothing,
        smoothing_size=args.smoothing_size,
        threshold_method=args.threshold,
        clip_limit=args.clip_limit,
        min_object_size=args.min_size,
        fill_holes=not args.no_fill_holes,
        apply_clahe_preprocessing=not args.no_clahe,
        save_outputs=not args.no_save,
        output_dir=args.output_dir
    )
    
    print(f"\n{'#'*60}")
    print(f"FINAL CONFLUENCY: {results['confluency']:.2f}%")
    print(f"OBJECTS DETECTED: {results['num_objects']}")
    print(f"{'#'*60}")
