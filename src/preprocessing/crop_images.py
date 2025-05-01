import cv2
import numpy as np
import os
import json
import copy
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


def process_all_images(top_percentage, bottom_percentage):
    """
    Process all images in input_dir and save cropped versions to output_dir.
    
    Args:
        top_percent: Percentage to crop from top (0-100)
        bottom_percent: Percentage to crop from bottom (0-100)
    """
    # Create output directory if it doesn't exist
    os.makedirs("data/images/cropped_images", exist_ok=True)
    
    # Get all image files from input directory
    image_files = [f for f in os.listdir("data/images/original_images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Cropping {top_percentage}% from top and {bottom_percentage}% from bottom")
    
    # Process each image
    for image_file in tqdm(image_files):
        input_path = os.path.join("data/images/original_images", image_file)
        output_path = os.path.join("data/images/cropped_images", image_file)
        
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error reading image: {input_path}")
            continue
        
        # Crop the image
        cropped_img = crop_percentage(img, top_percentage, bottom_percentage)
        
        # Save result
        cv2.imwrite(output_path, cropped_img)
    
    print(f"Cropped images saved to data/images/cropped_images")


def create_cropped_metadata(top_percentage, bottom_percentage):
    """
    Update metadata to reflect image cropping where top_percantage + bottom_percantage of pixels were removed from the image.
    
    Args:
        input_metadata_path: Path to the original metadata JSON
        output_metadata_path: Path to save the updated metadata
    """
    # Load the original metadata
    with open("data/images/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Original and new dimensions
    original_height = 3040
    new_height = original_height - int(original_height * (top_percentage + bottom_percentage) * 0.01)
    removed_top_pixels = int(original_height * top_percentage * 0.01)
    bottom_pixel = original_height - int(original_height * bottom_percentage * 0.01)


    # Statistics tracking
    updated_entries = 0
    updated_barcodes = 0
    skipped_barcodes = 0
    
    # Process each metadata entry
    updated_metadata = []
    
    for entry in metadata:
        # Create a deep copy to avoid modifying the original
        updated_entry = copy.deepcopy(entry)
        
        # Update camera height
        if "camera_info" in updated_entry and "left" in updated_entry["camera_info"]:
            updated_entry["camera_info"]["left"]["height"] = new_height
            
            # Update camera matrix - adjust principal point y (element at index 5)
            if "camera_matrix" in updated_entry["camera_info"]["left"]:
                camera_matrix = updated_entry["camera_info"]["left"]["camera_matrix"]
                # Principal point y is at index 5
                camera_matrix[5] = camera_matrix[5] - removed_top_pixels
            
            # Update projection matrix - adjust principal point y (element at index 7)
            if "projection_matrix" in updated_entry["camera_info"]["left"]:
                proj_matrix = updated_entry["camera_info"]["left"]["projection_matrix"]
                # Principal point y is at index 7
                proj_matrix[7] = proj_matrix[7] - removed_top_pixels
            
            updated_entries += 1
        
        # Update barcode coordinates if present
        if "barcodes" in updated_entry:
            updated_entry_copy = copy.deepcopy(updated_entry)
            for barcode in updated_entry_copy["barcodes"]:
                # Check if y coordinates will still be in bounds after adjustment
                y_pixel = barcode["y_pixel"]
                
                if y_pixel > bottom_pixel or y_pixel < removed_top_pixels:
                    # Remove barcode that would be out of bounds
                    skipped_barcodes += 1
                    updated_entry["barcodes"].remove(barcode)
                
                else:
                    # Adjust y coordinates
                    i = updated_entry["barcodes"].index(barcode)
                    barcode["top_left_y"] -= removed_top_pixels
                    barcode["top_right_y"] -= removed_top_pixels
                    barcode["bottom_left_y"] -= removed_top_pixels
                    barcode["bottom_right_y"] -= removed_top_pixels
                    barcode["y_pixel"] -= removed_top_pixels

                    updated_entry["barcodes"][i] = barcode
                    
                    
                updated_barcodes += 1

            
            # If all barcodes were out of bounds, entry may no longer have barcodes
            if not updated_entry["barcodes"]:
                del updated_entry["barcodes"]

        
        updated_metadata.append(updated_entry)
    
    # Save updated metadata
    with open("data/images/cropped_images_metadata.json", 'w') as f:
        json.dump(updated_metadata, f, indent=2)
    
    # Print summary
    print(f"Updated {updated_entries} metadata entries")
    print(f"Updated {updated_barcodes} barcodes")
    print(f"Skipped {skipped_barcodes} barcodes that would be out of bounds after cropping")
    
    return updated_metadata

if __name__ == "__main__":    
    top_percentage = 12
    bottom_percentage = 12
    # process_all_images(
    #     input_dir="data/images/original_images",
    #     output_dir="data/images/cropped_images",
    #     top_percent=top_percentage,
    #     bottom_percent=bottom_percentage
    # )
    create_cropped_metadata(top_percentage=top_percentage, bottom_percentage=bottom_percentage)