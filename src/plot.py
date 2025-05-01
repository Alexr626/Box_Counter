from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_count_boxes_results_comparison(result_file_path_1, result_file_path_2):
    images_per_bin_1 = {}
    accuracy_per_bin_1 = {}
    group_accuracy_per_bin_1 = {}
    mse_per_bin_1 = {}
    mae_per_bin_1 = {}
    images_per_bin_2 = {}
    accuracy_per_bin_2 = {}
    group_accuracy_per_bin_2 = {}
    mse_per_bin_2 = {}
    mae_per_bin_2 = {}

    df_1 = pd.read_csv(result_file_path_1)
    for index, row in df_1.iterrows():
        bin_id = row['bin_id']
        images_per_bin_1[bin_id] = row['num_images']
        accuracy_per_bin_1[bin_id] = row['accuracy']
        group_accuracy_per_bin_1[bin_id] = row['group_accuracy']
        mse_per_bin_1[bin_id] = row['mse']
        mae_per_bin_1[bin_id] = row['mae']

    df_2 = pd.read_csv(result_file_path_2)
    for index, row in df_2.iterrows():
        bin_id = row['bin_id']
        images_per_bin_2[bin_id] = row['num_images']
        accuracy_per_bin_2[bin_id] = row['accuracy']
        group_accuracy_per_bin_2[bin_id] = row['group_accuracy']
        mse_per_bin_2[bin_id] = row['mse']
        mae_per_bin_2[bin_id] = row['mae']

    # Group results by box group
    images_per_group_1, accuracy_per_group_1, mse_per_group_1, mae_per_group_1 = defaultdict(int), defaultdict(float), defaultdict(float), defaultdict(float)
    images_per_group_2, accuracy_per_group_2, mse_per_group_2, mae_per_group_2 = defaultdict(int), defaultdict(float), defaultdict(float), defaultdict(float)
    group_accuracy_per_group_1, group_accuracy_per_group_2 = defaultdict(float), defaultdict(float)
    with open('data/test_bins.csv', 'r') as file:
        test_bins = pd.read_csv(file)
        for index, row in test_bins.iterrows():
            bin_id = row['Bin_id']
            box_group = row['box_group']
            images_per_group_1[box_group] += images_per_bin_1[bin_id]
            accuracy_per_group_1[box_group] += accuracy_per_bin_1[bin_id] * images_per_bin_1[bin_id]
            group_accuracy_per_group_1[box_group] += group_accuracy_per_bin_1[bin_id] * images_per_bin_1[bin_id]
            mse_per_group_1[box_group] += mse_per_bin_1[bin_id] * images_per_bin_1[bin_id]
            mae_per_group_1[box_group] += mae_per_bin_1[bin_id] * images_per_bin_1[bin_id]

            images_per_group_2[box_group] += images_per_bin_2[bin_id]
            accuracy_per_group_2[box_group] += accuracy_per_bin_2[bin_id] * images_per_bin_2[bin_id]
            group_accuracy_per_group_2[box_group] += group_accuracy_per_bin_2[bin_id] * images_per_bin_2[bin_id]
            mse_per_group_2[box_group] += mse_per_bin_2[bin_id] * images_per_bin_2[bin_id]
            mae_per_group_2[box_group] += mae_per_bin_2[bin_id] * images_per_bin_2[bin_id]

        rmse_per_group_1 = {}
        for box_group in images_per_group_1:
            accuracy_per_group_1[box_group] /= images_per_group_1[box_group]
            group_accuracy_per_group_1[box_group] /= images_per_group_1[box_group]
            mse_per_group_1[box_group] /= images_per_group_1[box_group]
            rmse_per_group_1[box_group] = np.sqrt(mse_per_group_1[box_group])
            mae_per_group_1[box_group] /= images_per_group_1[box_group]

        rmse_per_group_2 = {}
        for box_group in images_per_group_2:
            accuracy_per_group_2[box_group] /= images_per_group_2[box_group]
            group_accuracy_per_group_2[box_group] /= images_per_group_2[box_group]
            mse_per_group_2[box_group] /= images_per_group_2[box_group]
            rmse_per_group_2[box_group] = np.sqrt(mse_per_group_2[box_group])
            mae_per_group_2[box_group] /= images_per_group_2[box_group]
    # Visualize the results
    plt.style.use('ggplot')
    bar_width = 0.35
    fig, axs = plt.subplots(2, 2, figsize=(6, 5))

    # Create x-axis positions for the bars
    x = np.arange(len(accuracy_per_group_1.keys()))
    
    # Define colors
    color_before = '#8cc5e3'  # Lighter blue
    color_after = '#1a80bb'   # Darker blue
    
    # Plot accuracy comparison
    axs[0, 0].bar(x - bar_width/2, accuracy_per_group_1.values(), width=bar_width, label='before FT', color=color_before)
    axs[0, 0].bar(x + bar_width/2, accuracy_per_group_2.values(), width=bar_width, label='after FT', color=color_after)
    axs[0, 0].set_xlabel('Bin group')
    axs[0, 0].set_ylabel('Box counting accuracy')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(accuracy_per_group_1.keys())
    axs[0, 0].legend()

    # Plot group accuracy comparison
    axs[0, 1].bar(x - bar_width/2, group_accuracy_per_group_1.values(), width=bar_width, label='before FT', color=color_before)
    axs[0, 1].bar(x + bar_width/2, group_accuracy_per_group_2.values(), width=bar_width, label='after FT', color=color_after)
    axs[0, 1].set_xlabel('Bin group')
    axs[0, 1].set_ylabel('Group classification accuracy')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(group_accuracy_per_group_1.keys())
    # axs[0, 1].legend()

    # Plot RMSE comparison
    axs[1, 0].bar(x - bar_width/2, rmse_per_group_1.values(), width=bar_width, label='before FT', color=color_before)
    axs[1, 0].bar(x + bar_width/2, rmse_per_group_2.values(), width=bar_width, label='after FT', color=color_after)
    axs[1, 0].set_xlabel('Bin group')
    axs[1, 0].set_ylabel('RMSE')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(rmse_per_group_1.keys())
    # axs[1, 0].legend()

    # Plot MAE comparison
    axs[1, 1].bar(x - bar_width/2, mae_per_group_1.values(), width=bar_width, label='before FT', color=color_before)
    axs[1, 1].bar(x + bar_width/2, mae_per_group_2.values(), width=bar_width, label='after FT', color=color_after)
    axs[1, 1].set_xlabel('Bin group')
    axs[1, 1].set_ylabel('MAE')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(mae_per_group_1.keys())
    # axs[1, 1].legend()

    fig.tight_layout()
    result_filename = result_file_path_1.split('/')[-1].split('.')[0]
    fig.savefig('results/'+result_filename+'_comparison.png', dpi=300)    


def plot_few_shot_results():
    n_shots = [0, 1, 2, 3, 4, 5]
    accuracy = [20.63, 30.34, 36.16, 37.21, 31.04, 38.98]
    group_accuracy = [44.44, 47.27, 46.08, 47.44, 47.27, 47.27]
    rmse = [6.65, 6.36, 5.00, 4.43, 4.95, 4.64]
    mae = [5.17, 4.70, 2.80, 2.54, 3.37, 2.80]
    
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=(6, 5))
    
    # Plot accuracy
    axs[0, 0].plot(n_shots, accuracy, marker='o', linestyle='-', linewidth=2)
    axs[0, 0].set_xlabel('Number of Shots')
    axs[0, 0].set_ylabel('Box counting accuracy')
    axs[0, 0].set_xticks(n_shots)
    # axs[0, 0].grid(True)
    
    # Plot group accuracy
    axs[0, 1].plot(n_shots, group_accuracy, marker='o', linestyle='-', linewidth=2)
    axs[0, 1].set_xlabel('Number of Shots')
    axs[0, 1].set_ylabel('Group classification accuracy')
    axs[0, 1].set_xticks(n_shots)
    # axs[0, 1].grid(True)
    
    # Plot RMSE
    axs[1, 0].plot(n_shots, rmse, marker='o', linestyle='-', linewidth=2)
    axs[1, 0].set_xlabel('Number of Shots')
    axs[1, 0].set_ylabel('RMSE')
    axs[1, 0].set_xticks(n_shots)
    # axs[1, 0].grid(True)
    
    # Plot MAE
    axs[1, 1].plot(n_shots, mae, marker='o', linestyle='-', linewidth=2)
    axs[1, 1].set_xlabel('Number of Shots')
    axs[1, 1].set_ylabel('MAE')
    axs[1, 1].set_xticks(n_shots)
    # axs[1, 1].grid(True)
    
    fig.tight_layout()
    fig.savefig('results/count_boxes_Qwen2_few_shot.png', dpi=300)

if __name__ == '__main__':
    plot_count_boxes_results_comparison('results/count_boxes_Qwen2.5-VL-7B-Instruct_prompt3.csv', 'results/count_boxes_Qwen2.5-VL-7B-Instruct_sft_1_prompt3_sft.csv')
    plot_few_shot_results()
    
    
