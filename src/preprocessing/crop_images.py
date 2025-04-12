import cv2
import numpy as np
import os
from tqdm import tqdm  # For progress bar

def crop_percentage(img, top_percent, bottom_percent):
    """
    Crop a fixed percentage from the top and bottom of an image.
    
    Args:
        img: The input image
        top_percent: Percentage to crop from top (0-100)
        bottom_percent: Percentage to crop from bottom (0-100)
        
    Returns:
        Cropped image
    """
    height, width = img.shape[:2]
    
    # Calculate crop boundaries
    top_crop = int(height * top_percent / 100)
    bottom_crop = int(height * (1 - bottom_percent / 100))
    
    # Ensure we don't have invalid crop dimensions
    if bottom_crop <= top_crop:
        print(f"Warning: Invalid crop dimensions. Using default values.")
        top_crop = int(height * 0.15)  # Default 15%
        bottom_crop = int(height * 0.85)  # Default 15%
    
    # Crop the image
    cropped_img = img[top_crop:bottom_crop, :]
    
    return cropped_img

def process_all_images(input_dir, output_dir, top_percent, bottom_percent):
    """
    Process all images in input_dir and save cropped versions to output_dir.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save cropped images
        top_percent: Percentage to crop from top (0-100)
        bottom_percent: Percentage to crop from bottom (0-100)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files from input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Cropping {top_percent}% from top and {bottom_percent}% from bottom")
    
    # Process each image
    for image_file in tqdm(image_files):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error reading image: {input_path}")
            continue
        
        # Crop the image
        cropped_img = crop_percentage(img, top_percent, bottom_percent)
        
        # Save result
        cv2.imwrite(output_path, cropped_img)
    
    print(f"Cropped images saved to {output_dir}")

def test_crop_percentages(input_path, percentages):
    """
    Test different cropping percentages on a single image and save for comparison.
    
    Args:
        input_path: Path to the input image
        percentages: List of (top_percent, bottom_percent) tuples to test
    """
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error reading image: {input_path}")
        return
    
    # Process for each percentage pair
    for i, (top_percent, bottom_percent) in enumerate(percentages):
        # Crop the image
        cropped_img = crop_percentage(img, top_percent, bottom_percent)
        
        # Create visualization image
        vis_img = img.copy()
        height, width = img.shape[:2]
        
        # Calculate and draw crop lines
        top_line = int(height * top_percent / 100)
        bottom_line = int(height * (1 - bottom_percent / 100))
        
        cv2.line(vis_img, (0, top_line), (width-1, top_line), (0, 255, 0), 2)
        cv2.line(vis_img, (0, bottom_line), (width-1, bottom_line), (0, 255, 0), 2)
        
        # Save results
        cv2.imwrite(f"debug_crop_{top_percent}_{bottom_percent}_vis.jpg", vis_img)
        cv2.imwrite(f"debug_crop_{top_percent}_{bottom_percent}_result.jpg", cropped_img)
    
    print(f"Test crops saved with prefixes 'debug_crop_'")

def process_single_image(input_path, top_percent, bottom_percent):
    """
    Process a single image and save debugging information.
    
    Args:
        input_path: Path to the input image
        top_percent: Percentage to crop from top (0-100)
        bottom_percent: Percentage to crop from bottom (0-100)
    """
    # Read image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error reading image: {input_path}")
        return
    
    # Crop the image
    cropped_img = crop_percentage(img, top_percent, bottom_percent)
    
    # Create visualization image
    vis_img = img.copy()
    height, width = img.shape[:2]
    
    # Calculate and draw crop lines
    top_line = int(height * top_percent / 100)
    bottom_line = int(height * (1 - bottom_percent / 100))
    
    cv2.line(vis_img, (0, top_line), (width-1, top_line), (0, 255, 0), 2)
    cv2.line(vis_img, (0, bottom_line), (width-1, bottom_line), (0, 255, 0), 2)
    
    # Save results
    cv2.imwrite("debug_original.jpg", img)
    cv2.imwrite("debug_cropped.jpg", cropped_img)
    cv2.imwrite("debug_vis.jpg", vis_img)
    
    print("Debug images saved")
    print(f"Cropped {top_percent}% from top and {bottom_percent}% from bottom")

if __name__ == "__main__":
    # Example usage
    
    # Option 1: Process a single image with fixed percentages
    process_single_image(
        input_path="data/images/original_images/1738952394744_587d51bf-dd41-4af2-a16e-39bc354fe881.jpg",
        top_percent=15,  # Crop 15% from top
        bottom_percent=15  # Crop 15% from bottom
    )
    
    # Option 2: Test multiple percentages on the same image
    test_crop_percentages(
        input_path="data/images/original_images/1738952394744_587d51bf-dd41-4af2-a16e-39bc354fe881.jpg",
        percentages=[
            (10, 10),  # Crop 10% from top and bottom
            (12, 12),  # Crop 10% from top and bottom
            (15, 15),  # Crop 15% from top and bottom
            (20, 20),  # Crop 20% from top and bottom
            (25, 25)   # Crop 25% from top and bottom
        ]
    )
    
    # Option 3: Process all images with fixed percentages
    process_all_images(
        input_dir="data/images/original_images",
        output_dir="data/images/cropped_images",
        top_percent=12,
        bottom_percent=12
    )