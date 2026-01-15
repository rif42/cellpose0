import os
import glob
import time
import numpy as np
from cellpose import models, io

def main():
    # Define paths
    base_dir = r"e:\work\cellpose-stemcell"
    # Process files in dataset/train and output to the same folder
    target_dir = os.path.join(base_dir, "dataset", "train")
    input_dir = target_dir
    output_dir = target_dir

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Model parameters
    # "run CPSAM" maps to the cpsam model
    # diameter:50, flow threshold:2, cellprob threshold:-0.5
    model_name = 'cpsam'
    diameter = 50
    flow_threshold = 2
    cellprob_threshold = -0.5
    niter=2000

    print(f"Loading model: {model_name}")
    # Initialize model
    model = models.CellposeModel(gpu=True, pretrained_model=model_name)

    # Get list of images recursively
    # Assuming common image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    image_files = []
    for ext in extensions:
        # Non-recursive glob since we are targeting a specific leaf directory
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Sort files for consistent processing order
    image_files.sort()

    print(f"Found {len(image_files)} images in {input_dir}")

    # Track results for summary report
    processing_results = []
    start_time = time.time()

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        # Save explicitly to the same directory
        output_subdir = output_dir
            
        print(f"Processing {img_name}...")
        
        try:
            # Read image
            img = io.imread(img_path)
            
            # Run evaluation
            masks, flows, styles = model.eval(
                img, 
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                niter=niter
            )

            # Save results as *_seg.npy
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_subdir, base_name)
            
            io.masks_flows_to_seg(img, masks, flows, output_path)
            
            # Count ROIs (number of unique masks, excluding background)
            num_rois = len(np.unique(masks)) - 1  # Subtract 1 for background (0)
            print(f"Saved result to {output_path}_seg.npy - Detected {num_rois} ROIs")
            
            # Store results
            processing_results.append({
                'filename': img_name,
                'roi_count': num_rois,
                'status': 'Success'
            })

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            processing_results.append({
                'filename': img_name,
                'roi_count': 0,
                'status': f'Error: {str(e)}'
            })

    # Generate summary report
    end_time = time.time()
    total_time = end_time - start_time
    
    # Create report filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"processing_report_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CELLPOSE BATCH PROCESSING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Parameters section
        f.write("PROCESSING PARAMETERS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Cell Diameter: {diameter}\n")
        f.write(f"Flow Threshold: {flow_threshold}\n")
        f.write(f"Cell Probability Threshold: {cellprob_threshold}\n")
        f.write(f"Input Directory: {input_dir}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Processing Time: {total_time:.2f} seconds\n\n")
        
        # Results section
        f.write("PROCESSING RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Filename':<30} {'ROI Count':<15} {'Status':<35}\n")
        f.write("-" * 80 + "\n")
        
        total_rois = 0
        successful_count = 0
        
        for result in processing_results:
            f.write(f"{result['filename']:<30} {result['roi_count']:<15} {result['status']:<35}\n")
            if result['status'] == 'Success':
                total_rois += result['roi_count']
                successful_count += 1
        
        # Summary statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Images Processed: {len(processing_results)}\n")
        f.write(f"Successfully Processed: {successful_count}\n")
        f.write(f"Failed: {len(processing_results) - successful_count}\n")
        f.write(f"Total ROIs Detected: {total_rois}\n")
        if successful_count > 0:
            f.write(f"Average ROIs per Image: {total_rois / successful_count:.2f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"Processing complete! Report saved to: {report_path}")
    print(f"Total images processed: {len(processing_results)}")
    print(f"Total ROIs detected: {total_rois}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
