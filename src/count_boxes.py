from src.dataset import create_hf_dataset, create_lazy_dataset, get_random_samples
from unsloth import FastVisionModel
from src.vision_utils import UnslothVisionDataCollator
import argparse
import csv
from collections import defaultdict
import re
import time
from dotenv import load_dotenv
import os
import numpy as np
from PIL import Image
from openai import OpenAI
import pandas as pd
from matplotlib import pyplot as plt
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import torch
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from .preprocessing.get_train_test_csvs import get_train_test_images
from .utils import encode_image, count2group
from .prompts import *

# Load environment variables from .env file
load_dotenv()

# Get the API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

def count_boxes_and_evaluate(images_directory: str, model_name: str, prompt: str, save: bool = True, cache: bool = True, comment: str = "", n_shots: int = 0):
    """
    Count the number of boxes in each image and evaluate the accuracy of the model.
    
    Args:
        images_directory (str): The directory containing the images
        model_name (str): The model to use to count the boxes
        prompt (str): The prompt fed to the model
        save (bool): Whether or not to save the results
        cache (bool): Whether or not to cache model responses
        comment (str): A comment to add to the results filename
    Returns:
        images_per_bin (dict): The number of images in each bin
        avg_count_per_bin (dict): The average count of the model in each bin
        accuracy_per_bin (dict): The accuracy of the model in each bin
        mse_per_bin (dict): The mean squared error of the model in each bin
        invalid_response_rate_per_bin (dict): The invalid response rate of the model in each bin
    """
    # Training set as few-shot examples
    train_ds = create_hf_dataset(images_directory)['train']
    example_set = create_lazy_dataset(train_ds, prompt)
    examples = get_random_samples(example_set, n_shots, seed=42)
    

    ground_truth_counts = {}
    groud_truth_group = {}
    with open('data/test_bins.csv', 'r') as file:
        test_bins = pd.read_csv(file)
        for index, row in test_bins.iterrows():
            bin_id = row['Bin_id']
            box_count_estimate = row['box_count_estimate']
            ground_truth_counts[bin_id] = box_count_estimate
            group = row['box_group']
            groud_truth_group[bin_id] = group

    images_per_bin = {}
    avg_count_per_bin = {}
    accuracy_per_bin = {}
    group_accuracy_per_bin = {}
    mse_per_bin = {}
    mae_per_bin = {}
    invalid_response_rate_per_bin = {}
    _, test_images = get_train_test_images(images_directory)
    num_test_images = sum([len(image_filenames) for image_filenames in test_images.values()])
    print(f"Counting boxes for {images_directory} with model {model_name}, {num_test_images} test images")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = False, # Set to False for 16bit LoRA
        max_seq_length = 4096
    )
    FastVisionModel.for_inference(model) # Enable for inference!

    model_name = model_name.split('/')[-1]

    # Count boxes for each image, group by bin
    start_time = time.time()
    for bin_id, image_filenames in tqdm(test_images.items(), desc="Processing bins", unit="bin"):
        box_count = ground_truth_counts[bin_id]
        count_pred = []

        if tokenizer:
            image_paths = [os.path.join(images_directory, image_filename) for image_filename in image_filenames]
            counts = count_boxes_unsloth(image_paths, prompt, model, tokenizer, examples)
            count_pred.extend(counts)
        else:
            for image_filename in tqdm(image_filenames, desc=f"Bin {bin_id}", leave=False, unit="img"):
                image_path = os.path.join(images_directory, image_filename)
                count = count_boxes_with_cache(image_path, model_name, prompt=prompt, comment=comment, write_to_cache=cache, read_from_cache=cache)
                count_pred.append(count)

        images_per_bin[bin_id] = len(count_pred)
        avg_count_per_bin[bin_id] = np.mean(count_pred)
        accuracy_per_bin[bin_id] = np.mean(np.array(count_pred) == box_count)
        group_accuracy_per_bin[bin_id] = np.mean(np.array(count2group(count_pred)) == groud_truth_group[bin_id])
        mse_per_bin[bin_id] = np.mean(np.abs(np.array(count_pred) - box_count)**2)
        mae_per_bin[bin_id] = np.mean(np.abs(np.array(count_pred) - box_count))
        invalid_response_rate_per_bin[bin_id] = np.mean(np.array(count_pred) == -1)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    num_images = np.array(list(images_per_bin.values()))
    num_images_total = np.sum(num_images)
    accuracies = np.array(list(accuracy_per_bin.values()))
    group_accuracies = np.array(list(group_accuracy_per_bin.values()))
    mses = np.array(list(mse_per_bin.values()))
    maes = np.array(list(mae_per_bin.values()))
    invalid_response_rates = np.array(list(invalid_response_rate_per_bin.values()))

    # Aggregate results of all bins
    accuracy_avg = np.dot(num_images, accuracies) / num_images_total
    group_accuracy_avg = np.dot(num_images, group_accuracies) / num_images_total
    rmse_avg = np.sqrt(np.dot(num_images, mses) / num_images_total)
    mae_avg = np.dot(num_images, maes) / num_images_total
    invalid_response_rate_avg = np.dot(num_images, invalid_response_rates) / num_images_total
    print(f"Accuracy: {accuracy_avg}")
    print(f"Group classificationaccuracy: {group_accuracy_avg}")
    print(f"RMSE: {rmse_avg}")
    print(f"MAE: {mae_avg}")
    print(f"Invalid response rate: {invalid_response_rate_avg}")
    print(f"Throughput: {num_images_total / (end_time - start_time)} images per second")

    # Save results grouped by bins to csv
    if save:
        if comment:
            result_filename = f'results/count_boxes_{model_name}_{comment}'
        else:
            result_filename = f'results/count_boxes_{model_name}'
        results_file = result_filename+'.csv'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as file:
            file.write(f"bin_id,num_images,avg_count,group_accuracy,accuracy,mse,mae,invalid_response_rate\n")
            for bin_id, num_images in images_per_bin.items():
                file.write(f"{bin_id},{num_images},{avg_count_per_bin[bin_id]},{group_accuracy_per_bin[bin_id]},{accuracy_per_bin[bin_id]},{mse_per_bin[bin_id]},{mae_per_bin[bin_id]},{invalid_response_rate_per_bin[bin_id]}\n")

        # Group results by box group
        images_per_group, accuracy_per_group, mse_per_group, mae_per_group, invalid_response_rate_per_group = defaultdict(int), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
        group_accuracy_per_group = defaultdict(float)
        with open('data/test_bins.csv', 'r') as file:
            test_bins = pd.read_csv(file)
            for index, row in test_bins.iterrows():
                bin_id = row['Bin_id']
                box_group = row['box_group']
                images_per_group[box_group] += images_per_bin[bin_id]
                accuracy_per_group[box_group] += accuracy_per_bin[bin_id] * images_per_bin[bin_id]
                group_accuracy_per_group[box_group] += group_accuracy_per_bin[bin_id] * images_per_bin[bin_id]
                mse_per_group[box_group] += mse_per_bin[bin_id] * images_per_bin[bin_id]
                mae_per_group[box_group] += mae_per_bin[bin_id] * images_per_bin[bin_id]
                invalid_response_rate_per_group[box_group] += invalid_response_rate_per_bin[bin_id] * images_per_bin[bin_id]
            
            print(f"Images per group: {images_per_group}")
            rmse_per_group = {}
            for box_group in images_per_group:
                accuracy_per_group[box_group] /= images_per_group[box_group]
                group_accuracy_per_group[box_group] /= images_per_group[box_group]
                mse_per_group[box_group] /= images_per_group[box_group]
                rmse_per_group[box_group] = np.sqrt(mse_per_group[box_group])
                mae_per_group[box_group] /= images_per_group[box_group]
                invalid_response_rate_per_group[box_group] /= images_per_group[box_group]

        # Visualize the results
        plt.style.use('ggplot')
        bar_width = 0.5
        fig, axs = plt.subplots(2, 2, figsize=(8, 7))

        axs[0, 0].bar(accuracy_per_group.keys(), accuracy_per_group.values(), width=bar_width)
        # axs[0, 0].set_title('Counting accuracy per group')
        axs[0, 0].set_xlabel('Box group')
        axs[0, 0].set_ylabel('Box counting accuracy')
        for i, (key, value) in enumerate(accuracy_per_group.items()):
            axs[0, 0].annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(0,1), ha='center')

        axs[0, 1].bar(group_accuracy_per_group.keys(), group_accuracy_per_group.values(), width=bar_width)
        # axs[0, 1].set_title('Group classification accuracy per group')
        axs[0, 1].set_xlabel('Box group')
        axs[0, 1].set_ylabel('Group classification accuracy')
        for i, (key, value) in enumerate(group_accuracy_per_group.items()):
            axs[0, 1].annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(0,1), ha='center')

        axs[1, 0].bar(rmse_per_group.keys(), rmse_per_group.values(), width=bar_width)
        # axs[1, 0].set_title('RMSE per group')
        axs[1, 0].set_xlabel('Box group')
        axs[1, 0].set_ylabel('RMSE')
        for i, (key, value) in enumerate(rmse_per_group.items()):
            axs[1, 0].annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(0,1), ha='center')

        axs[1, 1].bar(mae_per_group.keys(), mae_per_group.values(), width=bar_width)
        # axs[1, 1].set_title('MAE per group')
        axs[1, 1].set_xlabel('Box group')
        axs[1, 1].set_ylabel('MAE')
        for i, (key, value) in enumerate(mae_per_group.items()):
            axs[1, 1].annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(0,1), ha='center')

        fig.tight_layout()
        fig.savefig(result_filename+'.png', dpi=300)    


def count_boxes_with_cache(image_path: str, model_name: str, prompt: str, write_to_cache: bool = True, read_from_cache: bool = True, comment: str = "", **kwargs) -> int:
    """
    Count the number of boxes in an image using cache.

    Args:
        image_path (str): Path to the image file
        model_name (str): The model to use to count the boxes
        write_to_cache (bool): Whether to write the results to the cache
        read_from_cache (bool): Whether to read the results from the cache
        comment (str): A comment to add to the cache filename
    Returns:
        int: Number of boxes in the image
    """
    cache_file = f'cache/count_boxes_{model_name}.csv' if comment == "" else f'cache/count_boxes_{model_name}_{comment}.csv'
    
    if read_from_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as file:
                # Check if file is empty
                if os.path.getsize(cache_file) == 0:
                    print(f"Cache file {cache_file} is empty.")
                else:
                    reader = csv.reader(file)
                    # Skip the header line
                    next(reader)
                    for line in reader:
                        image, response_cached, count = line
                        if image_path == image:
                            # print(f"Found cached result for {image_path} with model {model}.")
                            return int(count)
        except Exception as e:
            print(f"Error reading cache file: {e}")
    
    if model_name == 'gpt':
        response = count_boxes_gpt(image_path, prompt)
    elif kwargs['model'] and kwargs['processor']:
        model, processor = kwargs['model'], kwargs['processor']
        response = count_boxes_vlm(image_path, prompt, model, processor)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    numbers = re.findall(r'\d+', response)
    count = -1
    if numbers:
        count = int(numbers[-1])
    else:
        print(f"Model response does not contain a number, returning -1")

    if write_to_cache:
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            with open(cache_file, 'a') as file:
                if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
                    file.write(f"image_path,response,count\n")
                writer = csv.writer(file)
                writer.writerow([image_path, response, count])
        except Exception as e:
            print(f"Error writing to cache file: {e}")

    return count
        

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def count_boxes_gpt(image_path: str, prompt: str) -> str:
    """
    Count the number of boxes in an image using ChatGPT API.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Response from the API
    """

    base64_image = encode_image(image_path)
    
    # Make the API call
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model='gpt-4o-mini',
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
    
    # Return the response text
    return response.output_text

def count_boxes_vlm(image_path: str, prompt: str, model, processor) -> str:
    """
    Count the number of boxes in an image using open source VLM.
    
    Args:
        image_path (str): Path to the image file
        model: The VLM model object
        processor: The VLM processor object
        
    Returns:
        str: Response of the model
    """

    

    # Prepare the prompt message
    # prompt = """Please count the number of boxes in this image. 
    # Return ONLY the number, nothing else. 
    # If you can't see any boxes, return 0."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])

    return output_text[0]

def count_boxes_unsloth(image_paths: list[str], prompt: str, model, tokenizer, examples: list[dict]) -> list[int]:
    """
    Count the number of boxes in a list of images using Unsloth fine-tuned model.
    
    Args:
        image_paths (list[str]): List of paths to the image files
        prompt (str): The prompt to use for the model
        model: The model object
        tokenizer: The tokenizer object
        examples: List of example messages and images for few-shot learning
    """
    counts = []
    def prepare_inference_data(image_path, prompt):
        """Prepare data in the same format as training data"""
        conversation = []
        for example in examples:
            conversation.extend(example['messages'])
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_path}
            ]
        })
        return {"messages": conversation}
    
    # Create data collator (same as training)
    data_collator = UnslothVisionDataCollator(model, tokenizer)

    # Process images in smaller batches to avoid token length issues
    batch_size = 8  # Adjust this based on your GPU memory
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing bins", unit="batch"):
        batch_paths = image_paths[i:i+batch_size]
        
        # Prepare data in training format
        data = [prepare_inference_data(image_path, prompt) for image_path in batch_paths]
        
        try:
            # Use collator to process the data
            inputs = data_collator(data).to("cuda")
            # print(f"input ids shape: {inputs['input_ids'].shape}")
            
            # Generate output
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
            )
            
            # Decode output
            text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for text_output in text_outputs:
                numbers = re.findall(r'\d+', text_output)
                count = -1
                if numbers:
                    count = int(numbers[-1])
                else:
                    print(f"Model response does not contain a number, returning -1")
                counts.append(count)
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Add placeholder counts for failed batch
            counts.extend([-1] * len(batch_paths))

    return counts


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_directory', type=str, default='data/original_images')
    parser.add_argument('--model', type=str, default='Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--prompt', type=str, default=prompt3)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--cache', type=bool, default=False)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--n_shots', type=int, default=0)   
    args = parser.parse_args()
    count_boxes_and_evaluate(images_directory=args.images_directory, model_name=args.model, prompt=args.prompt, save=args.save, cache=args.cache, comment=args.comment, n_shots=args.n_shots)
