import os
import pandas as pd
from datasets import Dataset, DatasetDict

from src.preprocessing.get_train_test_csvs import get_train_test_images
from src.utils import count2group


def create_hf_dataset(image_directory: str):
    """
    Create a Hugging Face dataset.
    
    Args:
        image_directory (str): Directory containing the images
    """
    ground_truth_counts = {}
    groud_truth_group = {}
    bins_file = f'data/bins.csv'
    with open(bins_file, 'r') as file:
        bins = pd.read_csv(file)
        for index, row in bins.iterrows():
            bin_id = row['Bin_id']
            box_count_estimate = row['box_count_estimate']
            ground_truth_counts[bin_id] = box_count_estimate
            group = count2group([box_count_estimate])[0]
            groud_truth_group[bin_id] = group

    # Create dataset dictionaries
    train_dataset_dict = {"image": [], "bin_id": [], "box_count_estimate": [], "box_group": []}
    test_dataset_dict = {"image": [], "bin_id": [], "box_count_estimate": [], "box_group": []}
    train_images, test_images = get_train_test_images(image_directory)

    for bin_id, image_filenames in train_images.items():
        for image_filename in image_filenames:
            image_path = os.path.join(image_directory, image_filename)
            train_dataset_dict["image"].append(image_path)
            train_dataset_dict["bin_id"].append(bin_id)
            train_dataset_dict["box_count_estimate"].append(ground_truth_counts[bin_id])
            train_dataset_dict["box_group"].append(groud_truth_group[bin_id])

    for bin_id, image_filenames in test_images.items():
        for image_filename in image_filenames:
            image_path = os.path.join(image_directory, image_filename)
            test_dataset_dict["image"].append(image_path)
            test_dataset_dict["bin_id"].append(bin_id)
            test_dataset_dict["box_count_estimate"].append(ground_truth_counts[bin_id])
            test_dataset_dict["box_group"].append(groud_truth_group[bin_id])

    # Create datasets
    train_dataset = Dataset.from_dict(train_dataset_dict)
    test_dataset = Dataset.from_dict(test_dataset_dict)
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    # Cast image column to Image type
    # dataset = dataset.cast_column('image', Image())
    
    print(f"Train split: {len(train_dataset)} examples")
    print(f"Test split: {len(test_dataset)} examples")
    
    return dataset

def convert_to_conversation(sample, prompt:str):
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : prompt},
                {"type" : "image", "image" : sample["image"]} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : sample["box_count_estimate"]} ]
            },
        ]
        return { "messages" : conversation }