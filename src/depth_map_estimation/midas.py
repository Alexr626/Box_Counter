# midas.py

import os
import sys
import subprocess
import json
import numpy as np
import torch
import cv2


def run_midas_depth_estimation(image_path, output_path, model_type="DPT_Large"):
    """
    Run MiDaS depth estimation on a single image
    
    Args:
        image_path: Path to input image
        output_path: Path to save the depth map (will be saved as .npy)
        model_type: One of "DPT_Large", "DPT_Hybrid", or "MiDaS_small"
    
    Returns:
        True if successful, False otherwise
    """

    import sys
    print("sys.path:", sys.path)

    # Get the path to the midas inference script
    midas_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "external_models", 
        "MiDaS"
    )
    
    # Add MiDaS to path temporarily
    sys.path.insert(0, midas_dir)
    
    # Import MiDaS modules (from the inserted path)
    from src.depth_map_estimation.MiDaS.midas.model_loader import load_model
    
    # Set up device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Set up model path
    weights_path = os.path.join(midas_dir, "weights")
    model_path = os.path.join(weights_path, "dpt_beit_large_512.pt")
    
    # Load model
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize=True)
    
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply transform
    img_input = transform({"image": img})["image"]
    
    # Compute depth
    with torch.no_grad():
        prediction = model.forward(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Get depth map
    depth_map = prediction.cpu().numpy()
    
    # Save depth map
    np.save(output_path, depth_map)
    
    # Also save a visualization
    vis_path = output_path.replace(".npy", "_vis.png")
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(vis_path, depth_vis)
    
    print(f"Depth map saved to {output_path}")
    print(f"Visualization saved to {vis_path}")
    
    # Remove the added path to avoid conflicts
    if midas_dir in sys.path:
        sys.path.remove(midas_dir)

    return True
    

def process_batch_with_midas(image_paths, output_dir, model_type="DPT_Large"):
    """Process a batch of images with MiDaS"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, image_path in enumerate(image_paths):
        # Create output filename based on input
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_depth.npy")
        
        print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
        success = run_midas_depth_estimation(image_path, output_path, model_type)
        
        results.append({
            "image_path": image_path,
            "depth_map_path": output_path if success else None,
            "success": success
        })
    
    # Save metadata about the processing
    with open(os.path.join(output_dir, "midas_processing_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    output_path = "path/to/output/depth.npy"
    run_midas_depth_estimation(image_path, output_path)