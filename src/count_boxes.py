import argparse
import re
import time
from dotenv import load_dotenv
import os
import numpy as np
from openai import OpenAI
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm

from .get_data import get_train_test_images
from .utils import encode_image

# Load environment variables from .env file
load_dotenv()

# Get the API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

def count_boxes_and_evaluate(images_directory: str, method: str = 'gpt'):
    """
    Count the number of boxes in each image and evaluate the accuracy of the model.
    
    Args:
        images_directory (str): The directory containing the images
        method (str): The method to use to count the boxes

    Returns:
        images_per_bin (dict): The number of images in each bin
        accuracy_per_bin (dict): The accuracy of the model in each bin
        avg_diff_per_bin (dict): The average difference between the predicted and ground truth counts in each bin
        invalid_response_rate_per_bin (dict): The invalid response rate of the model in each bin
    """
    ground_truth_counts = {}
    with open('data/test_bins.csv', 'r') as file:
        test_bins = pd.read_csv(file)
        for index, row in test_bins.iterrows():
            bin_id = row['Bin_id']
            box_count_estimate = row['box_count_estimate']
            ground_truth_counts[bin_id] = box_count_estimate

    images_per_bin = {}
    accuracy_per_bin = {}
    avg_diff_per_bin = {}
    invalid_response_rate_per_bin = {}
    _, test_images = get_train_test_images(images_directory)
    num_test_images = sum([len(image_filenames) for image_filenames in test_images.values()])
    print(f"Counting boxes for {num_test_images} images")

    for bin_id, image_filenames in tqdm(test_images.items(), desc="Processing bins", unit="bin"):
        box_count = ground_truth_counts[bin_id]
        count_pred = []
        count_gt = []

        for image_filename in tqdm(image_filenames, desc=f"Bin {bin_id}", leave=False, unit="img"):
            image_path = os.path.join(images_directory, image_filename)
            if method == 'gpt':
                count = count_boxes_gpt(image_path)
            else:
                raise NotImplementedError(f"Method {method} not implemented")
            count_pred.append(count)
            count_gt.append(box_count)

        images_per_bin[bin_id] = len(count_pred)
        accuracy_per_bin[bin_id] = np.mean(np.array(count_pred) == np.array(count_gt))
        avg_diff_per_bin[bin_id] = np.mean(np.abs(np.array(count_pred) - np.array(count_gt)))
        invalid_response_rate_per_bin[bin_id] = np.mean(np.array(count_pred) == -1)

    return images_per_bin, accuracy_per_bin, avg_diff_per_bin, invalid_response_rate_per_bin



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def count_boxes_gpt(image_path: str, model: str = "gpt-4o-mini", write_to_cache: bool = True, read_from_cache: bool = True) -> int:
    """
    Count the number of boxes in an image using ChatGPT API.
    
    Args:
        image_path (str): Path to the image file
        model (str): The model to use to count the boxes
        write_to_cache (bool): Whether to write the results to the cache
        read_from_cache (bool): Whether to read the results from the cache
        
    Returns:
        int: Number of boxes in the image
    """
    prompt = """Please count the number of boxes in this image. 
    Return ONLY the number, nothing else. 
    If you can't see any boxes, return 0."""
    
    cache_file = f'cache/count_boxes_{model}.csv'
    
    if read_from_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as file:
                # Check if file is empty
                if os.path.getsize(cache_file) == 0:
                    print(f"Cache file {cache_file} is empty.")
                else:
                    # Skip the header line
                    next(file)
                    for line in file:
                        image, response_cached, count = line.split(',')
                        if image_path == image:
                            print(f"Found cached result for {image_path} with model {model}.")
                            return int(count)
        except Exception as e:
            print(f"Error reading cache file: {e}")
                
    base64_image = encode_image(image_path)
    
    # Make the API call
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        # max_output_tokens=10
    )
    
    # Extract and return the number
    text = response.output_text
    # print(f"Model: {model}, response: {text}")
    number = re.search(r'\d+', text)
    count = -1
    if number:
        count = int(number.group())
    else:
        print(f"Model response does not contain a number, returning -1")
    
    if write_to_cache:
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            with open(cache_file, 'a') as file:
                if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
                    file.write(f"image_path,response,count\n")
                file.write(f"{image_path},{text},{count}\n")
        except Exception as e:
            print(f"Error writing to cache file: {e}")

    return count

def main(images_directory: str, method: str, save: bool = True):
    print(f"Counting boxes for {images_directory} with method {method}")

    start_time = time.time()
    images_per_bin, accuracy_per_bin, avg_diff_per_bin, invalid_response_rate_per_bin = count_boxes_and_evaluate(images_directory=images_directory, method=method)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    num_test_images = sum([num for num in images_per_bin.values()])
    accuracy_avg = np.mean(list(accuracy_per_bin.values()))
    avg_diff_avg = np.mean(list(avg_diff_per_bin.values()))
    invalid_response_rate_avg = np.mean(list(invalid_response_rate_per_bin.values()))
    print(f"Accuracy: {accuracy_avg}")
    print(f"Average difference: {avg_diff_avg}")
    print(f"Invalid response rate: {invalid_response_rate_avg}")
    print(f"Throughput: {num_test_images / (end_time - start_time)} images per second")

    if save:
        results_file = f'results/count_boxes_{method}.csv'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'a') as file:
            if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
                file.write(f"bin_id,num_images,accuracy_avg,avg_diff_avg,invalid_response_rate_avg\n")
            for bin_id, num_images in images_per_bin.items():
                file.write(f"{bin_id},{num_images},{accuracy_per_bin[bin_id]},{avg_diff_per_bin[bin_id]},{invalid_response_rate_per_bin[bin_id]}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_directory', type=str, default='data/original_images')
    parser.add_argument('--method', type=str, default='gpt')
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()
    main(images_directory=args.images_directory, method=args.method, save=args.save)
