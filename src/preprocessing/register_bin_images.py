import os
import re
import json
import cv2
import numpy as np
from get_data import group_photos_by_bin

def register_bin_images(original_images_directory):
    with open("data/metadata.json", "rb") as metadata:
        metadata_json = json.load(metadata)
    
    bin_to_image_dict = group_photos_by_bin(original_images_directory)
    registered_images = []

    for bin_id, bin_images in bin_to_image_dict.items():
        if len(bin_images) <= 1:
            continue  # Skip bins with only one image
        
        # Get metadata for the reference image (first image)
        ref_file = bin_images[0]
        ref_id = "-".join(re.split(r"[_\-\.]+", ref_file)[1:-1])  # Extract ID from filename
        ref_metadata = next(entry for entry in metadata_json if entry['_id'] == ref_id)
        
        # Get reference pose
        ref_rotation = convert_orientation_to_rotation(ref_metadata)
        ref_translation = extract_pose_position(ref_metadata)    

        for i, file in enumerate(bin_images):
            curr_id = "-".join(re.split(r"[_\-\.]+", file)[1:-1])
            curr_metadata = next(entry for entry in metadata_json if entry['_id'] == curr_id)
            
            # Skip the reference image (already in reference frame)
            if curr_id == ref_id:
                continue
                
            # Get current image pose
            curr_rotation = convert_orientation_to_rotation(curr_metadata)
            curr_translation = extract_pose_position(curr_metadata)
            
            # Calculate transformation from current image to reference frame
            # Note: You need to invert the reference transformation to get ref-to-world
            # Then compose with world-to-current to get ref-to-current
            # This is a simplification and might need adjustment based on your coordinate conventions
            rel_rvec, rel_tvec = calculate_relative_pose(
                ref_rotation, ref_translation, curr_rotation, curr_translation)
            
            # Store or use the transformation as needed
            registered_images.append({
                'image_path': os.path.join(original_images_directory, file),
                'bin_id': bin_id,
                'relative_rotation': rel_rvec,
                'relative_translation': rel_tvec
            })

    with open("data/registered_image_poses.json", "w") as file:
        serializable_images = numpy_to_json_serializable(registered_images)
        json.dump(serializable_images, file)


def numpy_to_json_serializable(obj):
    """Convert NumPy arrays to JSON serializable lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_json_serializable(item) for item in obj]
    else:
        return obj


def calculate_relative_pose(ref_rvec, ref_tvec, curr_rvec, curr_tvec):
    # Convert rotation vectors to matrices
    ref_rmat, _ = cv2.Rodrigues(ref_rvec)
    curr_rmat, _ = cv2.Rodrigues(curr_rvec)
    
    # Compute the relative transformation
    # If these are camera-to-world transformations:
    # We first invert the reference pose to get world-to-reference
    ref_rmat_inv = ref_rmat.T  # Transpose for rotation matrix inversion
    ref_tvec_inv = -np.dot(ref_rmat_inv, ref_tvec)
    
    # Then compose with the current pose to get reference-to-current
    rel_rmat = np.dot(curr_rmat, ref_rmat_inv)
    rel_tvec = curr_tvec + np.dot(curr_rmat, ref_tvec_inv)
    
    # Convert back to rotation vector
    rel_rvec, _ = cv2.Rodrigues(rel_rmat)

    rel_rvec = rel_rvec.flatten()
    
    return rel_rvec, rel_tvec

def extract_pose_position(json_list):
    return np.array([json_list['pose']['position']['x'], 
            json_list['pose']['position']['y'],
            json_list['pose']['position']['z']])

def extract_pose_quaternion(json_list):
    return [json_list['pose']['orientation']['x'],
            json_list['pose']['orientation']['y'],
            json_list['pose']['orientation']['z'],
            json_list['pose']['orientation']['w']]


def quaternion_to_rotation_matrix(q):
    # Assuming q is [w, x, y, z]
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,   2*x*z + 2*y*w],
        [2*x*y + 2*z*w,   1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x*x - 2*y*y]
    ])
    return R

def convert_orientation_to_rotation(metadata_entry):
    quaternion = extract_pose_quaternion(metadata_entry)
    R = quaternion_to_rotation_matrix(quaternion)
    
    rotation_vector, _ = cv2.Rodrigues(R)

    return rotation_vector
    

if __name__=="__main__":
    #group_photos_by_bin(original_images_directory="data/images/original_images")
    register_bin_images(original_images_directory="data/images/original_images")
