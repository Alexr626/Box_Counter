import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def read_pfm(file_path):
    """Read a PFM file into a numpy array."""
    with open(file_path, 'rb') as file:
        header = file.readline().decode('utf-8').rstrip()
        if header != 'PF':
            raise Exception('Not a PFM file.')
        
        dim_line = file.readline().decode('utf-8').rstrip()
        width, height = map(int, dim_line.split())
        
        scale_line = file.readline().decode('utf-8').rstrip()
        scale = float(scale_line)
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        
        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if header == 'PF' else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        
        return data, scale

def extract_depth_from_region(depth_map, x1, y1, x2, y2):
    """
    Extract depth statistics from a rectangular region in the depth map.
    
    Args:
        depth_map: The depth map (numpy array)
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
        
    Returns:
        Dictionary with depth statistics
    """
    # Ensure coordinates are within bounds
    height, width = depth_map.shape[:2]
    x1 = max(0, min(x1, width-1))
    x2 = max(0, min(x2, width-1))
    y1 = max(0, min(y1, height-1))
    y2 = max(0, min(y2, height-1))
    
    # Extract the region
    region = depth_map[y1:y2+1, x1:x2+1]
    
    # Calculate statistics
    stats = {
        'min_depth': float(np.min(region)),
        'max_depth': float(np.max(region)),
        'mean_depth': float(np.mean(region)),
        'median_depth': float(np.median(region)),
        'std_depth': float(np.std(region))
    }
    
    return stats

def visualize_depth_region(image_path, pfm_path, x1, y1, x2, y2, output_path=None):
    """
    Visualize the original image with the selected region highlighted,
    and display depth statistics for that region.
    
    Args:
        image_path: Path to the original RGB image
        pfm_path: Path to the PFM depth file
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
        output_path: Optional path to save the visualization
    """
    # Read the original image and depth map
    if image_path is not None and os.path.exists(image_path):
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        print("Original image not found or not provided.")
        original_img = None
    
    # Read the depth map
    try:
        depth_data, scale = read_pfm(pfm_path)
        # If depth_data has 3 channels, take the first one
        if len(depth_data.shape) > 2 and depth_data.shape[2] > 1:
            depth_data = depth_data[:,:,0]
    except Exception as e:
        print(f"Error reading PFM file: {e}")
        return
    
    # Normalize depth for visualization
    depth_vis = depth_data.copy()
    depth_vis = (depth_vis - np.min(depth_vis)) / (np.max(depth_vis) - np.min(depth_vis))
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
    
    # Extract depth statistics from the region
    stats = extract_depth_from_region(depth_data, x1, y1, x2, y2)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot depth map with region highlighted
    plt.subplot(1, 2, 1)
    plt.imshow(depth_vis)
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
    plt.title('Depth Map with Selected Region')
    
    # Plot original image with region highlighted (if available)
    plt.subplot(1, 2, 2)
    if original_img is not None:
        plt.imshow(original_img)
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
        plt.title('Original Image with Selected Region')
    else:
        # Extract the region from depth map for zoomed view
        region = depth_data[y1:y2+1, x1:x2+1]
        region_vis = (region - np.min(region)) / (np.max(region) - np.min(region))
        region_vis = (region_vis * 255).astype(np.uint8)
        region_vis = cv2.applyColorMap(region_vis, cv2.COLORMAP_INFERNO)
        region_vis = cv2.cvtColor(region_vis, cv2.COLOR_BGR2RGB)
        plt.imshow(region_vis)
        plt.title('Zoomed Region (Depth)')
    
    # Add depth statistics as text
    stats_text = "\n".join([f"{k}: {v:.4f}" for k, v in stats.items()])
    plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    plt.show()
    
    return stats

def analyze_multiple_images(image_dir, pfm_dir, regions, output_dir=None):
    """
    Analyze multiple images with predefined regions.
    
    Args:
        image_dir: Directory containing original images
        pfm_dir: Directory containing PFM depth files
        regions: Dictionary mapping image filenames to (x1,y1,x2,y2) coordinates
        output_dir: Directory to save visualizations
    
    Returns:
        Dictionary with depth statistics for each image/region
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for image_name, region in regions.items():
        # Construct paths
        image_path = os.path.join(image_dir, image_name)
        pfm_name = os.path.splitext(image_name)[0] + '.pfm'
        pfm_path = os.path.join(pfm_dir, pfm_name)
        
        # Skip if PFM file doesn't exist
        if not os.path.exists(pfm_path):
            print(f"PFM file not found for {image_name}, skipping.")
            continue
        
        # Set output path if needed
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_analysis.png")
        
        # Extract and visualize depth
        x1, y1, x2, y2 = region
        stats = visualize_depth_region(
            image_path if os.path.exists(image_path) else None,
            pfm_path, x1, y1, x2, y2, output_path
        )
        
        results[image_name] = stats
    
    return results

def convert_relative_depth_to_metric(depth_value, known_distance, known_depth_value):
    """
    Convert a relative depth value to metric distance.
    
    Args:
        depth_value: The depth value from MiDaS
        known_distance: A known real-world distance in meters
        known_depth_value: The corresponding depth value from MiDaS
    
    Returns:
        Estimated metric distance
    """
    # MiDaS depths are relative, so we use a simple scaling
    return (depth_value / known_depth_value) * known_distance

def compare_with_ground_truth(estimated_depths, ground_truth_depths, output_path=None):
    """
    Compare estimated depths with ground truth and visualize the results.
    
    Args:
        estimated_depths: Dictionary of estimated depths {image_name: depth}
        ground_truth_depths: Dictionary of ground truth depths {image_name: depth}
        output_path: Optional path to save the comparison plot
    """
    common_images = set(estimated_depths.keys()) & set(ground_truth_depths.keys())
    
    if not common_images:
        print("No common images to compare.")
        return
    
    # Extract data for plotting
    images = list(common_images)
    est_depths = [estimated_depths[img] for img in images]
    gt_depths = [ground_truth_depths[img] for img in images]
    
    # Calculate errors
    abs_errors = [abs(est - gt) for est, gt in zip(est_depths, gt_depths)]
    rel_errors = [abs(est - gt)/gt * 100 for est, gt in zip(est_depths, gt_depths)]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot estimated vs ground truth
    plt.subplot(2, 1, 1)
    plt.bar(range(len(images)), gt_depths, alpha=0.6, label='Ground Truth')
    plt.bar(range(len(images)), est_depths, alpha=0.6, label='Estimated')
    plt.xticks(range(len(images)), images, rotation=45, ha='right')
    plt.ylabel('Depth (m)')
    plt.title('Estimated vs Ground Truth Depths')
    plt.legend()
    
    # Plot relative errors
    plt.subplot(2, 1, 2)
    plt.bar(range(len(images)), rel_errors)
    plt.xticks(range(len(images)), images, rotation=45, ha='right')
    plt.ylabel('Relative Error (%)')
    plt.title('Depth Estimation Relative Error')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Comparison saved to {output_path}")
    
    plt.show()
    
    # Calculate overall statistics
    mean_abs_error = np.mean(abs_errors)
    mean_rel_error = np.mean(rel_errors)
    max_rel_error = np.max(rel_errors)
    
    print(f"Mean Absolute Error: {mean_abs_error:.4f} m")
    print(f"Mean Relative Error: {mean_rel_error:.2f}%")
    print(f"Max Relative Error: {max_rel_error:.2f}%")

# Example usage:
if __name__ == "__main__":
    # Example 1: Extract depth from a single region in a single image
    pfm_path = "path/to/your/depth_map.pfm"
    # Region of interest (horizontal orange bar)
    x1, y1, x2, y2 = 100, 200, 500, 230  # Example coordinates - adjust to your image
    
    # Extract depth statistics
    stats = extract_depth_from_region(read_pfm(pfm_path)[0], x1, y1, x2, y2)
    print("Depth statistics for the region:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    
    # Example 2: Visualize the region and depth statistics
    image_path = "path/to/your/original_image.jpg"  # Optional
    visualize_depth_region(image_path, pfm_path, x1, y1, x2, y2, "depth_region_analysis.png")
    
    # Example 3: Analyze multiple images with different regions
    regions = {
        "image1.jpg": (100, 200, 500, 230),
        "image2.jpg": (120, 210, 520, 240),
        # Add more images and regions as needed
    }
    results = analyze_multiple_images(
        "path/to/original/images",
        "path/to/pfm/files",
        regions,
        "path/to/output/visualizations"
    )
    
    # Example 4: Compare with ground truth if available
    # This requires calibrating the relative depths to metric distances
    # using at least one known measurement
    
    # First convert to metric using a reference measurement
    known_distance = 1.5  # meters
    known_depth_value = results["reference_image.jpg"]["mean_depth"]
    
    estimated_metric_depths = {}
    for img, stats in results.items():
        depth = convert_relative_depth_to_metric(
            stats["mean_depth"], known_distance, known_depth_value)
        estimated_metric_depths[img] = depth
    
    # Ground truth depths from your metadata
    ground_truth_depths = {
        "image1.jpg": 1.2,  # meters
        "image2.jpg": 1.8,  # meters
        # Add all ground truth values
    }
    
    # Compare and visualize
    compare_with_ground_truth(
        estimated_metric_depths, 
        ground_truth_depths,
        "depth_comparison.png"
    )
    