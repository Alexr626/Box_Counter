import cv2
import numpy as np
import os
import json
import copy
from tqdm import tqdm  # For progress bar
from preprocessing.preprocess_raw_data import create_train_test_bin_sets, group_photos_by_bin
from preprocessing.crop_images import process_all_images, create_cropped_metadata
from preprocessing.register_bin_images import register_bin_images
from depth_map_estimation.depth_estimations import get_depth_estimations
from depth_map_estimation.extract_pixel_depths import propagate_depth_calibration_for_vertical_flight, calculate_barcode_metadata


# Creates subdirectory of cropped images in data/ and created metadata file that reflects cropping
# Bottom and top percentages of rows of pixel to crop set to 12 to git rid of black areas due to wideview camera
def crop_images(bottom_percentage = 12, top_percentage = 12):
    process_all_images(top_percentage=top_percentage,
                       bottom_percentage=bottom_percentage)
    
    create_cropped_metadata(top_percentage=top_percentage,
                                 bottom_percentage=bottom_percentage)
    

# Does necessary preprocessing, including train/test set creation, calculating the depths of barcodes in images, and registering all images to 
def preprocess_data(with_cropped_images=True):
    create_train_test_bin_sets()
    register_bin_images(with_cropped_images)


def main(depth_model, with_cropped_images = True):
    if depth_model == "midas":
        model_id = "Intel/dpt-hybrid-midas"
    else:
        model_id = "LiheYoung/depth-anything-small-hf"

    # if with_cropped_images:
    #     crop_images()

    # preprocess_data(with_cropped_images)
    
    get_depth_estimations(model_id=model_id, 
                          model_name=depth_model)

    propagate_depth_calibration_for_vertical_flight(depth_model=depth_model,
                                                    with_cropped_images=with_cropped_images)

if __name__=="__main__":
    main(depth_model="depth_anything")