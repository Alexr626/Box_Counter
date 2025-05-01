import numpy as np
import os
import json
from scipy.spatial.transform import Rotation
from scipy import optimize
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import open3d as o3d


def convert_to_absolute_depth(relative_depth, params):
    """
    Convert relative depth to absolute metric depth.
    
    Args:
        relative_depth: MiDaS relative depth map
        params: Calibration parameters with a_param, b_param, and scale_factor
        
    Returns:
        array: Absolute depth map in meters
    """
    # Create a cleaned copy of the depth map
    depth = np.copy(relative_depth)
    
    # Replace invalid values with median
    valid_mask = (depth > 0) & np.isfinite(depth)
    if np.sum(valid_mask) > 0:
        median_depth = np.median(depth[valid_mask])
        depth[~valid_mask] = median_depth
    else:
        # If no valid values, return zeros
        return np.zeros_like(depth)
    
    # Apply calibration
    if params.get('a_param') is not None and params.get('b_param') is not None:
        # Use the complete formula: true_depth = a * (1/relative_depth) + b
        a = params['a_param']
        b = params['b_param']
        
        # Avoid division by zero
        inv_depth = np.zeros_like(depth)
        inv_depth[depth > 1e-10] = 1.0 / depth[depth > 1e-10]
        
        absolute_depth = a * inv_depth + b
        
        # Clip negative values
        absolute_depth = np.maximum(absolute_depth, 0)
    elif params.get('scale_factor') is not None:
        # Fallback to simple scaling
        absolute_depth = depth * params['scale_factor']
    else:
        # If no calibration available, return the original (but this won't be metric)
        absolute_depth = depth
    
    return absolute_depth

def depth_to_point_cloud(depth_map, camera_matrix, mask=None):
    """
    Convert a depth map to a point cloud in camera coordinates.
    
    Args:
        depth_map: Absolute depth map in meters
        camera_matrix: Camera intrinsics matrix
        mask: Optional binary mask of pixels to include
        
    Returns:
        tuple: (points, colors) arrays
    """
    # Extract camera intrinsics
    fx = camera_matrix[0]
    fy = camera_matrix[4]
    cx = camera_matrix[2]
    cy = camera_matrix[5]
    
    # Create pixel coordinate grid
    height, width = depth_map.shape
    v, u = np.mgrid[0:height, 0:width]
    
    # Apply mask if provided
    if mask is not None:
        valid_mask = mask & (depth_map > 0)
    else:
        valid_mask = depth_map > 0
    
    # Extract valid pixels
    z = depth_map[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    
    # Back-project to 3D
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy
    
    # Stack into points array
    points = np.column_stack((x, y, z))
    
    # Create simple grayscale colors based on depth
    norm_depth = (z - np.min(z)) / (np.max(z) - np.min(z)) if np.max(z) > np.min(z) else np.zeros_like(z)
    colors = np.column_stack((norm_depth, norm_depth, norm_depth))
    
    return points, colors

def transform_points_to_world(points, position, orientation):
    """
    Transform points from camera to world coordinates.
    
    Args:
        points: Array of 3D points in camera coordinates
        position: Camera position (x, y, z)
        orientation: Camera orientation quaternion (x, y, z, w)
        
    Returns:
        array: Points in world coordinates
    """
    # Extract camera position
    camera_pos = np.array([position["x"], position["y"], position["z"]])
    
    # Create rotation matrix from quaternion
    quat = [orientation["x"], orientation["y"], orientation["z"], orientation["w"]]
    rotation = Rotation.from_quat(quat)
    rotation_matrix = rotation.as_matrix()
    
    # Transform points
    world_points = np.zeros_like(points)
    for i in range(len(points)):
        # Apply rotation
        rotated = rotation_matrix @ points[i]
        # Apply translation
        world_points[i] = rotated + camera_pos
    
    return world_points


def create_bin_point_clouds(calibration_params, metadata_file, bin_coordinates_file, 
                           depth_map_dir, output_dir):
    """
    Process all images to create point clouds for each bin.
    
    Args:
        calibration_params: Depth calibration parameters for each image
        metadata_file: Path to metadata JSON
        bin_coordinates_file: Path to bin coordinates JSON
        depth_map_dir: Directory containing depth maps
        output_dir: Directory to save output point clouds
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata and bin coordinates
    with open(metadata_file, 'r') as f:
       metadata = json.load(f)
    
    with open(bin_coordinates_file, 'r') as f:
        bin_coordinates = json.load(f)
    
    # Group images by bin
    bin_images = defaultdict(list)
    
    for image_path, params in calibration_params.items():
        bin_id = params["bin_id"]
        
        # Extract image ID from path
        image_id = os.path.basename(image_path).split('_')[1].split('.')[0]
        
        # Skip if no metadata
        if image_id not in metadata:
            continue
            
        bin_images[bin_id].append({
            "image_id": image_id,
            "image_path": image_path,
            "calibration": params
        })
    
    # Process each bin
    bin_stats = {}
    
    for bin_id, images in bin_images.items():
        print(f"Processing bin {bin_id} with {len(images)} images...")
        
        # Skip bins without coordinates
        if bin_id not in bin_coordinates:
            print(f"  No coordinates for bin {bin_id}, skipping")
            continue
            
        # Get bin boundaries
        x_min, z_min, x_max, z_max = bin_coordinates[bin_id]
        
        # Initialize combined point cloud
        all_points = []
        all_colors = []
        
        # Create visualization folder
        bin_viz_dir = os.path.join(output_dir, f"bin_{bin_id}_viz")
        os.makedirs(bin_viz_dir, exist_ok=True)
        
        # Track image counts by confidence level
        confidence_counts = {
            'high': 0,
            'medium': 0,
            'low': 0,
            'very_low': 0
        }
        
        # Process each image for this bin
        for i, img_data in enumerate(images):
            image_id = img_data["image_id"]
            calibration = img_data["calibration"]
            img_data
            
            # Track confidence
            confidence = calibration.get('confidence', 'low')
            confidence_counts[confidence] += 1
            
            # Get metadata
            if image_id not in metadata:
                continue
                
            meta = metadata[image_id]
            
            # # Find depth map file
            # depth_map_dir = os.path.join("data/depth", depth_model) 
            # depth_file = os.path.join(depth_map_dir, f"{timestamp}_{image_id}-{depth_model}.pfm")
                
            # Read relative depth map
            relative_depth = read_pfm(depth_file)
            
            # Convert to absolute depth
            absolute_depth = convert_to_absolute_depth(relative_depth, calibration)
            
            # Save a visualization of the conversion
            if i % 5 == 0:  # Save every 5th image to avoid clutter
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 2, 1)
                plt.imshow(relative_depth, cmap='viridis')
                plt.colorbar(label='Relative Depth')
                plt.title('Relative Depth')
                
                plt.subplot(1, 2, 2)
                plt.imshow(absolute_depth, cmap='viridis')
                plt.colorbar(label='Absolute Depth (m)')
                plt.title(f'Absolute Depth (Confidence: {confidence})')
                
                plt.suptitle(f"Bin {bin_id} - Image {image_id}")
                plt.tight_layout()
                plt.savefig(os.path.join(bin_viz_dir, f"depth_{image_id}.png"))
                plt.close()
            
            # Get camera parameters
            camera_matrix = meta["camera_info"]["left"]["camera_matrix"]
            position = meta["pose"]["position"]
            orientation = meta["pose"]["orientation"]
            
            # Convert to point cloud in camera coordinates
            points, colors = depth_to_point_cloud(absolute_depth, camera_matrix)
            
            # Skip if no points
            if len(points) == 0:
                continue
            
            # Transform to world coordinates
            world_points = transform_points_to_world(points, position, orientation)
            
            # Filter points within bin boundaries
            mask = (
                (world_points[:, 0] >= x_min) & (world_points[:, 0] <= x_max) &
                (world_points[:, 2] >= z_min) & (world_points[:, 2] <= z_max)
            )
            
            # Apply additional reasonability filter - keep only points within reasonable range
            y_min = min([img["calibration"]["y_position"] for img in images]) - 2
            y_max = max([img["calibration"]["y_position"] for img in images]) + 2
            
            mask = mask & (world_points[:, 1] >= y_min) & (world_points[:, 1] <= y_max)
            
            # Apply distance filter - remove points that are very far from camera
            point_distances = np.linalg.norm(world_points - np.array([position["x"], position["y"], position["z"]]), axis=1)
            max_distance = 10.0  # Maximum reasonable distance in meters
            mask = mask & (point_distances <= max_distance)
            
            filtered_points = world_points[mask]
            filtered_colors = colors[mask]
            
            # Weight points by confidence
            confidence_weight = {
                'high': 1.0,
                'low': 0.6,
                'very_low': 0.3
            }.get(confidence, 0.5)
            
            # Add to combined point cloud
            if len(filtered_points) > 0:
                all_points.append(filtered_points)
                all_colors.append(filtered_colors * confidence_weight)
        
        # Skip if no points found
        if not all_points:
            print(f"  No valid points for bin {bin_id}")
            continue
            
        # Combine points
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # Downsample to reduce noise and size
        voxel_size = 0.02  # 2cm voxel size
        downsampled = pcd.voxel_down_sample(voxel_size)
        
        # Remove outliers
        cleaned, _ = downsampled.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        
        # Save point cloud
        output_path = os.path.join(output_dir, f"{bin_id}_pointcloud.ply")
        o3d.io.write_point_cloud(output_path, cleaned)
        
        # Calculate statistics
        num_points = len(cleaned.points)
        
        # Save statistics
        bin_stats[bin_id] = {
            "num_points": num_points,
            "num_images_used": len(images),
            "confidence_counts": confidence_counts,
            "bin_dimensions": {
                "width": x_max - x_min,
                "height": z_max - z_min
            },
            "point_cloud_path": output_path
        }
        
        print(f"  Created point cloud with {num_points} points")
        
        # Create a final visualization of the point cloud
        if num_points > 0:
            try:
                # Create a simple visualization
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(cleaned)
                
                # Set view
                view_control = vis.get_view_control()
                view_control.set_front([0, 0, -1])
                view_control.set_up([0, 1, 0])
                
                # Capture image
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(os.path.join(bin_viz_dir, f"{bin_id}_pointcloud.png"))
                vis.destroy_window()
            except Exception as e:
                print(f"  Could not create visualization: {e}")
    
    # Save statistics
    with open(os.path.join(output_dir, "bin_statistics.json"), 'w') as f:
        json.dump(bin_stats, f, indent=2)
    
    return bin_stats

# Example usage of the full pipeline
def run_full_pipeline(metadata_file, registered_images_file, bin_coordinates_file, 
                      depth_map_dir, output_dir):
    """
    Run the complete pipeline from metadata to point clouds.
    
    Args:
        metadata_file: Path to metadata JSON
        registered_images_file: Path to registered images JSON
        bin_coordinates_file: Path to bin coordinates JSON
        depth_map_dir: Directory containing depth maps
        output_dir: Base directory for outputs
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Propagate depth calibration
    print("Step 2: Propagating depth calibration...")
    calibration_params = propagate_depth_calibration_for_vertical_flight(
        metadata_file,
        registered_images_file,
        os.path.join(output_dir, "barcode_depths.json"),
        os.path.join(output_dir, "depth_calibration.json"),
        depth_map_dir
    )
    
    # Step 3: Create bin point clouds
    print("Step 3: Creating point clouds...")
    bin_stats = create_bin_point_clouds(
        calibration_params,
        metadata_file,
        bin_coordinates_file,
        depth_map_dir,
        os.path.join(output_dir, "point_clouds")
    )
    
    print("Pipeline complete!")
    return bin_stats