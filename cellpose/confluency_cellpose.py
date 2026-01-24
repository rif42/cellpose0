"""
Cell Confluency Calculator using Cellpose

This script takes an image, calculates the cell probability map using Cellpose,
applies thresholding to create a binary mask (live cells vs background/contaminants),
and calculates the cell confluency (percentage of image area covered by cells).

Based on Cellpose's multi-stage filtering approach:
1. Semantic Rejection: The network assigns low probability logits to texture-mismatched debris
2. Topological Rejection: flow consistency check discards objects without coherent center-seeking flows
3. Morphological Rejection: min_size parameter eliminates small components

The cell probability map (cellprob) is a logit value where:
- Logit > 0: High confidence cell (probability > 0.5)
- Logit < 0: High confidence background (probability < 0.5)
- Logit = 0: 50% probability threshold
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cellpose import models, io




def sigmoid(x):
    """
    Convert logits to probabilities using the sigmoid function.
    
    p = 1 / (1 + exp(-x))
    
    - Logit 0 → 0.5 (50% probability)
    - Logit +6 → ~0.9975 (high confidence cell)
    - Logit -6 → ~0.0025 (high confidence background)
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def get_probability_map(cellprob_logits):
    """
    Convert the cell probability logit map to actual probabilities (0-1 range).
    
    Args:
        cellprob_logits: Raw cell probability output from Cellpose (logits)
    
    Returns:
        Probability map with values in [0, 1] range
    """
    return sigmoid(cellprob_logits)


def calculate_confluency(
    image_path: str,
    model_name: str = 'cpsam',
    diameter: float = 50,
    flow_threshold: float = 2.0,
    cellprob_threshold: float = -1,
    niter: int = 2000,
    use_gpu: bool = True,
    save_outputs: bool = True,
    output_dir: str = None
) -> dict:
    """
    Calculate cell confluency from an image using Cellpose.
    Outputs visualization images at each processing step.
    
    The confluency is calculated by:
    1. Running Cellpose to get the cell probability map
    2. Applying probability threshold to create binary mask
    3. Calculating the percentage of the image covered by cells
    
    Args:
        image_path: Path to the input image
        model_name: Cellpose model to use ('cyto', 'cyto2', 'nuclei', 'cpsam', etc.)
        diameter: Expected cell diameter in pixels
        flow_threshold: Flow error threshold for rejecting bad masks (default 2.0)
        cellprob_threshold: Cell probability threshold (default 0.0 = 50% probability)
        niter: Number of iterations for flow dynamics (default 2000)
        use_gpu: Whether to use GPU acceleration
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
    # STEP 1: Load and display original image
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: Loading Original Image")
    print(f"{'='*60}")
    
    img = io.imread(image_path)
    print(f"Loaded image: {image_path}")
    
    # Get image dimensions
    if img.ndim == 2:
        height, width = img.shape
        channels = 1
        img_display = img
    elif img.ndim == 3:
        height, width, channels = img.shape
        img_display = img
    else:
        raise ValueError(f"Unexpected image dimensions: {img.ndim}")
    
    total_pixels = height * width
    print(f"Image size: {width}x{height} ({channels} channel{'s' if channels > 1 else ''})")
    
    # Apply CLAHE preprocessing
    print("Applying CLAHE preprocessing...")
    
    # Convert to grayscale if needed
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Create CLAHE object and apply
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    
    # Use CLAHE-processed image for Cellpose
    img = img_clahe
    img_display = img_clahe  # For visualization
    print("CLAHE preprocessing complete")    
    
    # =========================================================================
    # STEP 2: Run Cellpose and get Cell Probability Map
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Computing Cell Probability Map")
    print(f"{'='*60}")
    
    print(f"Loading Cellpose model: {model_name}")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_name)
    
    print("Running Cellpose segmentation...")
    masks, flows, styles = model.eval(
        img,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        niter=niter
    )
    
    # Extract cell probability map (logits)
    cellprob_map = flows[2]
    
    # Convert to probability (0-1 range) for visualization
    probability_map = get_probability_map(cellprob_map)
    
    print(f"Cell probability map stats (logits):")
    print(f"  Min: {cellprob_map.min():.3f}")
    print(f"  Max: {cellprob_map.max():.3f}")
    print(f"  Mean: {cellprob_map.mean():.3f}")
    
    
    # =========================================================================
    # STEP 3: Apply threshold to create Binary Mask
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: Creating Binary Mask (Cells vs Background)")
    print(f"{'='*60}")
    
    # Apply threshold to logits
    binary_mask = cellprob_map > cellprob_threshold
    
    # Calculate cell area
    cell_area_pixels = np.sum(binary_mask)
    confluency = (cell_area_pixels / total_pixels) * 100
    
    print(f"Threshold applied: {cellprob_threshold} (logit)")
    print(f"Equivalent probability: {sigmoid(cellprob_threshold):.2%}")
    print(f"Cell pixels: {cell_area_pixels:,} / {total_pixels:,}")
    print(f"Confluency: {confluency:.2f}%")
    
    
    
    # =========================================================================
    # FINAL: Create side-by-side comparison of all steps
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL: Creating Side-by-Side Process Visualization")
    print(f"{'='*60}")
    
    if save_outputs:
        final_path = os.path.join(output_dir, f"{base_name}_cellpose_overview.png")
        
        # Create 1x3 horizontal layout for the 3 steps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Step 1: Original Image
        ax1 = axes[0]
        if img.ndim == 2:
            ax1.imshow(img, cmap='gray')
        else:
            ax1.imshow(img)
        ax1.set_title('Step 1: Original Image', fontsize=14, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Step 2: Probability Map
        ax2 = axes[1]
        im2 = ax2.imshow(probability_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title('Step 2: Cell Probability Map', fontsize=14, fontweight='bold', pad=10)
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Probability', fontsize=10)
        
        # Step 3: Binary Mask
        ax3 = axes[2]
        cmap_binary = LinearSegmentedColormap.from_list('binary_cells', ['black', '#00FF00'])
        ax3.imshow(binary_mask, cmap=cmap_binary)
        ax3.set_title(f'Step 3: Binary Mask\nConfluency: {confluency:.1f}%', 
                      fontsize=14, fontweight='bold', pad=10)
        ax3.axis('off')
        
        # Add overall title with results
        fig.suptitle(f'Cell Confluency Analysis Pipeline\n'
                     f'Model: {model_name} | Diameter: {diameter} | '
                     f'Threshold: {cellprob_threshold}',
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(final_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {final_path}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("CONFLUENCY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Cell probability threshold: {cellprob_threshold}")
    print(f"Total image area: {total_pixels:,} pixels")
    print(f"Cell area: {cell_area_pixels:,} pixels")
    print(f"Cell confluency: {confluency:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'confluency': confluency,
        'cell_area_pixels': int(cell_area_pixels),
        'total_area_pixels': int(total_pixels),
        'cellprob_map': cellprob_map,
        'probability_map': probability_map,
        'binary_mask': binary_mask
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate cell confluency from an image using Cellpose'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='cpsam',
        help='Cellpose model to use (default: cpsam)'
    )
    parser.add_argument(
        '--diameter', '-d',
        type=float,
        default=50,
        help='Expected cell diameter in pixels (default: 50)'
    )
    parser.add_argument(
        '--flow_threshold', '-ft',
        type=float,
        default=2.0,
        help='Flow error threshold (default: 2.0)'
    )
    parser.add_argument(
        '--cellprob_threshold', '-ct',
        type=float,
        default=-1,
        help='Cell probability threshold (default: 0.0 = 50%% probability)'
    )
    parser.add_argument(
        '--niter',
        type=int,
        default=2000,
        help='Number of flow dynamics iterations (default: 2000)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: same as input image)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output files'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Run confluency calculation
    results = calculate_confluency(
        image_path=args.image_path,
        model_name=args.model,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        niter=args.niter,
        use_gpu=not args.cpu,
        save_outputs=not args.no_save,
        output_dir=args.output_dir
    )
    
    print(f"\n{'#'*60}")
    print(f"FINAL CONFLUENCY: {results['confluency']:.2f}%")
    print(f"{'#'*60}")
