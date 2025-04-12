import json
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_test_bin_sets():
    bins_df = pd.read_csv('data/bins.csv')

    # Create a simpler stratification variable that won't create too many small groups
    # Just use box count ranges which should be most important for your task
    bins_df['box_group'] = pd.cut(
        bins_df['box_count_estimate'], 
        bins=[-0.01, 0.99, 4.99, 14.99, 100], # empty ∈ [0, 1), few ∈ [1, 5), medium ∈ [5, 15), many = 15+
        labels=['empty', 'few', 'medium', 'many']
    )

    # Check counts to make sure we don't have too small groups
    print(bins_df['box_group'].value_counts())

    # Split based on just box count groups
    train_bins, test_bins = train_test_split(
        bins_df, 
        test_size=0.2, 
        random_state=42,
        stratify=bins_df['box_group']
    )

    # Verify distributions manually
    print(f"\nTraining set: {len(train_bins)} bins")
    print(f"Test set: {len(test_bins)} bins")

    # Check box count distribution
    print("\nBox count distribution:")
    print("Original:", bins_df['box_group'].value_counts(normalize=True))
    print("Train:", train_bins['box_group'].value_counts(normalize=True))
    print("Test:", test_bins['box_group'].value_counts(normalize=True))

    # Save the split
    train_bins.to_csv('data/train_bins.csv', index=False)
    test_bins.to_csv('data/test_bins.csv', index=False)

    # Save bin IDs for easy reference in your code
    train_bin_ids = train_bins['Bin_id'].tolist()
    test_bin_ids = test_bins['Bin_id'].tolist()

    with open('data/train_bin_ids.txt', 'w') as f:
        f.write('\n'.join(train_bin_ids))
        
    with open('data/test_bin_ids.txt', 'w') as f:
        f.write('\n'.join(test_bin_ids))

def get_train_test_images(original_images_directory):
    with open('data/train_bin_ids.txt', 'r') as file:
        train_bins = [line.strip() for line in file]

    with open('data/test_bin_ids.txt', 'r') as file:
        test_bins = [line.strip() for line in file]

    bin_to_image_dict = group_photos_by_bin(original_images_directory)
    
    train_images = {}
    test_images = {}

    for train_bin in train_bins:
        train_images[train_bin] = bin_to_image_dict[train_bin]

    for test_bin in test_bins:
        test_images[test_bin] = bin_to_image_dict[test_bin]

    return train_images, test_images
    



def group_photos_by_bin(original_images_directory, save=False):
    with open("data/metadata.json", "rb") as metadata_file:
        metadata_json = json.load(metadata_file)
    
    bin_to_image_dict = {}
    
    # Count files for debugging
    file_count = 0
    match_count = 0
    
    for root, dirs, files in os.walk(original_images_directory):
        for file in files:
            file_count += 1
            
            # Extract the UUID using regex
            uuid_pattern = r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
            match = re.search(uuid_pattern, file)
            
            if match:
                image_id = match.group(1)
                
                # Find matching metadata entry
                matching_entries = [entry for entry in metadata_json if entry['_id'] == image_id]
                
                if matching_entries:
                    match_count += 1
                    metadata_entry = matching_entries[0]
                    
                    # Extract bin ID
                    if 'bin_id' in metadata_entry and metadata_entry['bin_id']:
                        curr_bin_id = metadata_entry['bin_id'][0]
                        
                        # Initialize bin list if it doesn't exist
                        if curr_bin_id not in bin_to_image_dict:
                            bin_to_image_dict[curr_bin_id] = []
                        
                        # Append file to the bin
                        bin_to_image_dict[curr_bin_id].append(file)

    if save:
        with open("data/bin_image_groups.json", "w") as file:
            json.dump(bin_to_image_dict, file)
        
    return bin_to_image_dict


if __name__=="__main__":
    create_train_test_bin_sets()
    get_train_test_images("data/images/original_images")