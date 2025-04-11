import os
import sys
import json
import numpy as np
import torch
import cv2

def run_midas_depth_estimation(image_path, output_path, model_type="DPT_Large"):
    """
    Run MiDaS depth estimation on a single image using the torch.hub interface.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the depth map as a .npy file.
        model_type (str): One of "DPT_Large", "DPT_Hybrid", or "MiDaS_small".
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Set up device; use GPU if available.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Load the MiDaS model from the official repository using torch.hub.
        # The first argument specifies the repo and the second selects the model.
        midas = torch.hub.load("isl-org/MiDaS", model_type)
        midas.to(device)
        midas.eval()
        
        # Load the transforms offered by the MiDaS repo.
        transforms = torch.hub.load("isl-org/MiDaS", "transforms")
        # Depending on the model, select the appropriate transform.
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            transform = transforms.dpt_transform
        else:  # For "MiDaS_small"
            transform = transforms.small_transform
        
        # Load image with OpenCV and convert from BGR to RGB.
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found or unable to load: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transformation and add a batch dimension.
        input_image = transform(img).unsqueeze(0).to(device)
        
        # Run inference.
        with torch.no_grad():
            prediction = midas(input_image)
            # Upsample the prediction to the original image resolution.
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert prediction to a NumPy array.
        depth_map = prediction.cpu().numpy()
        
        # Save the depth map as a .npy file.
        np.save(output_path, depth_map)
        
        # Save a visualization: normalize, apply a colormap, and write out.
        vis_path = output_path.replace(".npy", "_vis.png")
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(vis_path, depth_vis)
        
        print(f"Depth map saved to {output_path}")
        print(f"Visualization saved to {vis_path}")
        return True
        
    except Exception as e:
        print(f"Error running MiDaS: {e}")
        return False

def process_batch_with_midas(image_paths, output_dir, model_type="DPT_Large"):
    """Process a batch of images with MiDaS."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, image_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_depth.npy")
        print(f"Processing {i + 1}/{len(image_paths)}: {image_path}")
        success = run_midas_depth_estimation(image_path, output_path, model_type)
        
        results.append({
            "image_path": image_path,
            "depth_map_path": output_path if success else None,
            "success": success
        })
    
    # Save a JSON file summarizing the processing results.
    results_path = os.path.join(output_dir, "midas_processing_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Example usage.
    # Replace these paths with valid paths on your system.
    image_path = "path/to/your/image.jpg"
    output_path = "path/to/output/depth.npy"
    run_midas_depth_estimation(image_path, output_path)
