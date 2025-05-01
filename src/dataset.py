import os
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
import torch

from src.preprocessing.get_train_test_csvs import get_train_test_images
from src.utils import count2group

class LazyImageDataset(TorchDataset):
    def __init__(self, image_paths, box_counts, prompt):
        self.image_paths = image_paths
        self.box_counts = box_counts
        self.prompt = prompt

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        box_count = self.box_counts[idx]

        # with Image.open(image_path) as image:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image", "image": image_path}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": str(box_count)}
            ]}
        ]
        return {
            "messages": messages,
            # "images": [image],
        }

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
    # test_dataset_dict = {"image": [], "bin_id": [], "box_count_estimate": [], "box_group": []}
    train_images, test_images = get_train_test_images(image_directory)

    for bin_id, image_filenames in train_images.items():
        for image_filename in image_filenames:
            image_path = os.path.join(image_directory, image_filename)
            train_dataset_dict["image"].append(image_path)
            train_dataset_dict["bin_id"].append(bin_id)
            train_dataset_dict["box_count_estimate"].append(ground_truth_counts[bin_id])
            train_dataset_dict["box_group"].append(groud_truth_group[bin_id])
    # for bin_id, image_filenames in test_images.items():
    #     for image_filename in image_filenames:
    #         image_path = os.path.join(image_directory, image_filename)
    #         test_dataset_dict["image"].append(image_path)
    #         test_dataset_dict["bin_id"].append(bin_id)
    #         test_dataset_dict["box_count_estimate"].append(ground_truth_counts[bin_id])
    #         test_dataset_dict["box_group"].append(groud_truth_group[bin_id])

    # Create datasets
    train_dataset = Dataset.from_dict(train_dataset_dict)
    # test_dataset = Dataset.from_dict(test_dataset_dict)
    dataset = DatasetDict({
        'train': train_dataset,
        # 'test': test_dataset
    })
    
    # Cast image column to Image type
    # dataset = dataset.cast_column('image', Image())
    
    print(f"Train split: {len(train_dataset)} examples")
    # print(f"Test split: {len(test_dataset)} examples")
    
    return dataset

def create_lazy_dataset(dataset, prompt: str):
    """
    Create a lazy-loading dataset from a Hugging Face dataset.
    
    Args:
        dataset: Hugging Face dataset
        prompt (str): The prompt to use for the model
    """
    return LazyImageDataset(
        image_paths=dataset["image"],
        box_counts=dataset["box_count_estimate"],
        prompt=prompt
    )

def convert_to_conversation(sample, prompt:str):
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : prompt},
                {"type" : "image", "image" : sample["image"]} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : str(sample["box_count_estimate"])} ]
            },
        ]
        return { "messages" : conversation }

def format_messages_and_images(sample, prompt: str):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": str(sample["box_count_estimate"])}
        ]}
    ]
    image = Image.open(sample["image"])
    width, height = image.size
    if width > height:
        new_width = 1024
        new_height = int(height * (1024 / width))
    else:
        new_height = 1024
        new_width = int(width * (1024 / height))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return {
        "messages": messages,
        "images": [image]
    }

def get_random_samples(dataset, n_samples: int, seed: int = None):
    """
    Randomly select n samples from a PyTorch dataset.
    
    Args:
        dataset: PyTorch dataset
        n_samples (int): Number of samples to select
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        list: List of randomly selected samples
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Get random indices
    indices = torch.randperm(len(dataset))[:n_samples]
    
    # Get samples
    samples = [dataset[i] for i in indices]
    
    return samples
