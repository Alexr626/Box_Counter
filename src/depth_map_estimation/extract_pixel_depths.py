import numpy as np
import os
import json
from scipy.spatial.transform import Rotation
from scipy import optimize
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import open3d as o3d
from preprocessing.preprocess_raw_data import group_photos_by_bin

def propagate_depth_calibration_for_vertical_flight(depth_model,
                                                    with_cropped_images=True):
    """
    Propagate depth calibration specifically for vertical drone flights.
    
    Args:
        metadata_file: Path to the full metadata JSON
        registered_images_file: Path to the registered images data
        barcode_depths_file: Path to the barcode depth references
        output_file: Path to save calibration parameters for all images
        depth_map_dir: Directory containing depth map files
    """    
    bin_image_groups = group_photos_by_bin(save=True)
    
    # Group images by bin
    try:
        with open(f"data/depth/{depth_model}/barcode_depths_{depth_model}.json", 'r') as f:
            barcode_depths = defaultdict(json.load(f))

    except Exception as e:
        barcode_depths = calculate_barcode_metadata(depth_model=depth_model,
                                                    with_cropped_images=with_cropped_images)


    # Mark images with barcode depths and calculate depth parameters
    print("Calculating depth parameters for images with barcodes...")

    try:
        barcode_depths_with_parameterization_file = f"data/depth/{depth_model}/barcode_depths_with_parameterization_{depth_model}.json"
        with open(barcode_depths_with_parameterization_file, 'r') as f:
            depth_calibrations = json.load(f)
    except Exception as e:
        print(e)
        depth_calibrations = calculate_depth_parameters_all_images(barcode_depths,
                                                                   depth_model=depth_model)
        

    # Now propagate parameters to images without barcodes
    print("Propagating depth parameters to all images...")
    
    for bin_id, image_ids in bin_image_groups.items():
        # Skip if bin has no images
        if not image_ids:
            continue
            
        # Find images with reliable depth parameters in this bin
        reference_images = []
        for img_id in image_ids:
            if img_id in depth_calibrations:
                img_data = depth_calibrations[img_id]
                if (img_data.get("has_multiple_barcodes", False) and 
                    "depth_estimation_parameters" in img_data and
                    img_data["depth_estimation_parameters"].get("a_param") is not None and
                    img_data["depth_estimation_parameters"].get("b_param") is not None):
                    reference_images.append(img_data)
        
        # Skip bins with no reference images
        if not reference_images:
            continue
            
        # Process each image in this bin
        for img_id in image_ids:
            if img_id not in depth_calibrations:
                continue
                
            img_data = depth_calibrations[img_id]
            
            # Skip if already has valid parameters
            if (img_data.get("depth_estimation_parameters", {}).get("a_param") is not None and
                img_data.get("depth_estimation_parameters", {}).get("b_param") is not None):
                # Already has parameters, add confidence if not present
                if "depth_estimation_parameters" in img_data:
                    img_data["depth_estimation_parameters"]["confidence"] = "high"
                continue
                
                
            # Propagate parameters based on nearby reference images
            y_pos = img_data.get("y_position")
            
            if y_pos is not None and reference_images:
                # Get reference positions and values
                ref_positions = np.array([r.get("y_position") for r in reference_images])
                ref_a_values = np.array([r.get("depth_estimation_parameters", {}).get("a_param") for r in reference_images])
                ref_b_values = np.array([r.get("depth_estimation_parameters", {}).get("b_param") for r in reference_images])
                
                # Find nearest neighbors
                dists = np.abs(ref_positions - y_pos)
                nearest_idx = np.argsort(dists)[:min(3, len(dists))]
                
                # Weight by inverse distance
                weights = 1.0 / (dists[nearest_idx] + 1e-6)
                weights = weights / np.sum(weights)
                
                # Weighted average
                a_param = float(np.sum(weights * ref_a_values[nearest_idx]))
                b_param = float(np.sum(weights * ref_b_values[nearest_idx]))
                
                # Update parameters
                img_data["depth_estimation_parameters"]["a_param"] = a_param
                img_data["depth_estimation_parameters"]["b_param"] = b_param
                img_data["depth_estimation_parameters"]["confidence"] = "interpolated"
    
    # Find global average for any remaining images without parameters
    valid_params = [data["depth_estimation_parameters"] for data in depth_calibrations.values() 
                   if "depth_estimation_parameters" in data and 
                   data["depth_estimation_parameters"].get("a_param") is not None and
                   data["depth_estimation_parameters"].get("b_param") is not None and
                   data["depth_estimation_parameters"].get("confidence") == "high"]
    
    if valid_params:
        global_a = np.median([p["a_param"] for p in valid_params])
        global_b = np.median([p["b_param"] for p in valid_params])
        
        # Apply global average to remaining images
        for img_id, img_data in depth_calibrations.items():
            if ("depth_estimation_parameters" not in img_data or
                img_data["depth_estimation_parameters"].get("a_param") is None or
                img_data["depth_estimation_parameters"].get("b_param") is None):
                
                # Initialize if needed
                if "depth_estimation_parameters" not in img_data:
                    img_data["depth_estimation_parameters"] = {}
                
                # Set global parameters
                img_data["depth_estimation_parameters"]["a_param"] = float(global_a)
                img_data["depth_estimation_parameters"]["b_param"] = float(global_b)
                img_data["depth_estimation_parameters"]["confidence"] = "global_average"
    
    # Save updated barcode depths
    output_file = f"data/depth/{depth_model}/barcode_depths_with_calibration_{depth_model}.json"
    with open(output_file, 'w') as f:
        json.dump(depth_calibrations, f, indent=2)
    
    print(f"Updated barcode depths file with propagated parameters: {output_file}")
    return depth_calibrations


def calculate_barcode_metadata(depth_model, with_cropped_images=True):
    """
    Calculate the absolute 3D distance from camera to each detected barcode and include
    additional information for depth parameter calculation.
    
    Args:
        with_cropped_images: Whether to use cropped image metadata
        depth_model: The depth model name used for depth maps
        
    Returns:
        dict: Enhanced barcode depth references
    """
    if with_cropped_images:
        metadata_path = "data/images/cropped_images_metadata.json"
    else:
        metadata_path = "data/images/metadata.json"
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Initialize results container
    barcode_metadata = {}
    
    # Process each metadata entry
    for entry in metadata:
        # Handle different possible metadata formats
        image_id = entry.get("_id")
        bin_ids = entry.get("bin_id")
        timestamp = entry["header"]["capture_time"]
        y_position = entry["pose"]["position"]["y"]
        
        # Build file paths
        img_path = f"data/images/{timestamp}_{image_id}.jpg"
        depth_path = f"data/depth/{depth_model}/depth_maps_and_estimates/{timestamp}_{image_id}-{depth_model}.pfm"
        
        camera_pos = np.array([
            entry["pose"]["position"]["x"],
            entry["pose"]["position"]["y"],
            entry["pose"]["position"]["z"]
        ])
        
        # Process barcodes if they exist
        image_barcode_depths = []
        has_multiple_barcodes = False
        
        if "barcodes" in entry and entry["barcodes"]:
            barcodes = entry["barcodes"]
            has_multiple_barcodes = len(barcodes) > 1
            
            for barcode in barcodes:
                # Get 3D world position of barcode
                barcode_pos = np.array([
                    barcode["x_world"],
                    barcode["y_world"],
                    barcode["z_world"]
                ])
                
                # Calculate Euclidean distance from camera to barcode
                distance = np.linalg.norm(camera_pos - barcode_pos)
                
                # Record depth reference
                reference = {
                    "x_pixel": barcode["x_pixel"],
                    "y_pixel": barcode["y_pixel"],
                    "world_position": barcode_pos.tolist(),
                    "absolute_depth": distance,
                    "barcode_id": barcode["barcode"]
                }
                image_barcode_depths.append(reference)
        
        # Store all information for this image
        barcode_metadata[image_id] = {
            "bin_ids": bin_ids,
            "timestamp": timestamp,
            "image_path": img_path,
            "depth_path": depth_path,
            "y_position": y_position,
            "has_multiple_barcodes": has_multiple_barcodes,
            "barcode_depths": image_barcode_depths,
            "camera_position": camera_pos.tolist()
        }
    
    # Save depth references
    output_file = f"data/depth/{depth_model}/barcode_depths_{depth_model}.json"
    with open(output_file, 'w') as f:
        json.dump(barcode_metadata, f, indent=2)
    print(f"Saved barcode depths to {output_file}")
    
    return barcode_metadata


def calculate_depth_parameters_all_images(barcode_depths,
                                          depth_model):
    """
    Calculate depth parameters for all images and add them to the barcode depths file.
    
    Args:
        depth_model: Depth model used for depth map generation
        
    Returns:
        dict: Updated barcode depths with depth estimation parameters
    """
    # Process each image
    for image_id, image_data in barcode_depths.items():
        # Check if we have enough barcodes for this image
        if image_data["has_multiple_barcodes"]:
            # Calculate depth parameters
            a_param, b_param, error = calculate_depth_params_for_image(image_data)
            
            # Store calculated parameters
            barcode_depths[image_id]["depth_estimation_parameters"] = {
                "a_param": a_param,
                "b_param": b_param,
                "error": error,
            }
        else:
            # Not enough barcode references
            barcode_depths[image_id]["depth_estimation_parameters"] = {
                "a_param": None,
                "b_param": None,
                "error": "insufficient_barcodes",
            }
    
    # Save updated barcode depths
    barcode_depths_with_parameters_file = f"data/depth/{depth_model}/barcode_depths_with_parameterization_{depth_model}.json"
    with open(barcode_depths_with_parameters_file, 'w') as f:
        json.dump(barcode_depths, f, indent=2)
    print(f"Updated barcode depths with depth parameters in {barcode_depths_with_parameters_file}")
    
    return barcode_depths


def calculate_depth_params_for_image(image_data):
    """
    Calculate the A and B parameters for depth conversion for an image.
    
    Args:
        image_id: The ID of the image
        barcode_depths: Dictionary of barcode depth references
        depth_model: The depth model used for depth maps
        
    Returns:
        tuple: (A, B, error, simple_scale_factor)
    """
    # Get image data from barcode_depths
    barcode_refs = image_data["barcode_depths"]
    depth_file = image_data["depth_path"]

    # Load depth map
    
    depth_map = read_pfm(depth_file)
    try: 
        sample = depth_map[50, 50]
    
    except Exception as e:
        print(e)
        return None, None, None

    # Collect relative depths and true depths
    inverse_relative_depths = []
    true_depths = []
    
    for ref in barcode_refs:
        # Get pixel coordinates
        x_pixel = ref["x_pixel"]
        y_pixel = ref["y_pixel"]
        
        # Get absolute depth from barcode reference
        absolute_depth = ref["absolute_depth"]
        
        # Get predicted relative depth from depth map
        inverse_relative_depth = depth_map[y_pixel, x_pixel]
            
        # Only use valid depth values
        if np.isfinite(inverse_relative_depth):
            inverse_relative_depths.append(inverse_relative_depth)
            true_depths.append(absolute_depth)

    inverse_relative_depths = np.array(inverse_relative_depths)
    true_depths = np.array(true_depths)
    
    a, b, error = calculate_depth_conversion_params(inverse_relative_depths, true_depths)

    return a, b, error



def read_pfm(file_path):
    """Read a PFM file and return the depth map."""
    try:
        with open(file_path, 'rb') as file:
            # Read header
            header = file.readline().decode('utf-8').strip()
            size = file.readline().decode('utf-8').strip().split()
            width, height = int(size[0]), int(size[1])
            scale = float(file.readline().decode('utf-8').strip())
            
            # Read data
            buffer = file.read()
            data = np.frombuffer(buffer, dtype=np.float32)
            data = data.reshape(height, width)
            
            # Handle endianness
            if scale < 0:
                data = data.byteswap()
            
            return data
    except Exception as e:
        print(f"Couldn't read pfm file: {e}")

        return None


def calculate_depth_conversion_params(inverse_relative_depths, true_depths):
    """
    Calculate the scale (A) and shift (B) parameters for depth conversion.
    
    Args:
        relative_depths: Array of relative depth values from MiDaS
        true_depths: Array of corresponding true depth values
        
    Returns:
        tuple: (A, B) parameters and fit error
    """
    if len(inverse_relative_depths) < 2 or len(true_depths) < 2:
        return None, None, 'inf'

    
    # Convert relative depths to inverse relative depths
    relative_depths = 1.0 / np.clip(inverse_relative_depths, 1e-10, None)
    # relative_depths = np.clip(inverse_relative_depths, 1e-10, None)
    
    # Check if we have valid values
    if not np.all(np.isfinite(relative_depths)):
        return None, None, 'inf'
    
    # Define error function for optimization
    def error_function(params, x, y):
        a, b = params
        y_pred = a * x + b
        return np.sum((y - y_pred)**2)
    
    # Initial guess based on linear regression
    x_mean = np.mean(relative_depths)
    y_mean = np.mean(true_depths)
    numerator = np.sum((relative_depths - x_mean) * (true_depths - y_mean))
    denominator = np.sum((relative_depths - x_mean)**2)
    
    if denominator != 0:
        a_initial = numerator / denominator
        b_initial = y_mean - a_initial * x_mean
    else:
        a_initial = 1.0
        b_initial = 0.0
    
    initial_params = [a_initial, b_initial]
    
    try:
        # Perform optimization to find A and B
        result = optimize.minimize(
            error_function, 
            initial_params, 
            args=(relative_depths, true_depths),
            method='L-BFGS-B'
        )
        
        if result.success:
            a, b = result.x
            # Calculate fit error (RMSE)
            y_pred = a * relative_depths + b
            # y_pred = 1.0 / (a * relative_depths + b)
            rmse = np.sqrt(np.mean((true_depths - y_pred)**2))
            return float(a), float(b), float(rmse)
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Fallback to direct linear regression
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(relative_depths.reshape(-1, 1), true_depths)
        a = model.coef_[0]
        b = model.intercept_
        
        # Calculate fit error (RMSE)
        y_pred = a * relative_depths + b
        rmse = np.sqrt(np.mean((true_depths - y_pred)**2))
        return float(a), float(b), float(rmse)
    except Exception as e:
        print(f"Linear regression failed: {e}")
        
    return None, None, 'inf'


# if __name__=="__main__":
#     metadata_file="data/images/cropped_images_metadata.json"
#     bin_coordinates_file = "data/bin_coordinates.json"

#     calibration_params = propagate_depth_calibration_for_vertical_flight(metadata_file="data/images/cropped_images_metadata.json",
#                                                     registered_images_file="data/images/registered_image_poses.json",
#                                                     barcode_depths_file="data/barcode_depths.json",
#                                                     output_file="data/barcode_depths_with_calibration_depth_anything.json",
#                                                     depth_model="depth_anything")
    
# Usage example:
# run_full_pipeline(
#     "metadata.json",
#     "registered_images.json",
#     "bin_coordinates.json",
#     "depth_directory",
#     "output_directory"
# )