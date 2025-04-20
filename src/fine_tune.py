import json
import os

import wandb
from src.dataset import convert_to_conversation, create_hf_dataset
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import torch

from .preprocessing.get_train_test_csvs import get_train_test_images
from .utils import count2group
from .prompts import *

LORA_RANK = 32
NUM_EPOCHS = 1
LR = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 2048

def fine_tune_qwen(model_name: str, image_directory: str, prompt: str, comment: str):
    # Load the model
    model, tokenizer = FastVisionModel.from_pretrained(
        "Qwen/" + model_name,
        load_in_4bit = False,
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )

    output_dir = "results/" + model_name + "_" + comment

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )

    # Construct the dataset
    ds = create_hf_dataset(image_directory)
    train_ds = ds['train']
    converted_dataset = [convert_to_conversation(sample, prompt) for sample in train_ds]
    # print(converted_dataset[0])

    # # Test model inference
    # FastVisionModel.for_inference(model)

    # image = Image.open(test_ds[2]["image"])
    # messages = [
    #     {"role": "user", "content": [
    #         {"type": "image"},
    #         {"type": "text", "text": prompt}
    #     ]}
    # ]
    # input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    # inputs = tokenizer(
    #     image,
    #     input_text,
    #     add_special_tokens = False,
    #     return_tensors = "pt",
    # ).to("cuda")
    # generated_ids = model.generate(**inputs, max_new_tokens = 128,
    #                use_cache = True, temperature = 1.5, min_p = 0.1)
    # output = tokenizer.decode(generated_ids[0], skip_special_tokens = True)
    # print(output)

    # Fine-tune the model
    FastVisionModel.for_training(model) # Enable for training!

    # Use wandb for logging
    wandb.init(
        project='Box_counter',
        job_type="training",
        anonymous="allow",
        name=model_name + "_" + comment,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = converted_dataset,
        args = SFTConfig(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            warmup_steps = 5,
            num_train_epochs = NUM_EPOCHS, # Set this instead of max_steps for full training runs
            learning_rate = LR,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "wandb",     # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = MAX_LENGTH,
        ),
    )
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()
    wandb.finish()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save hyperparameters
    hyperparameters = {
        "num_epochs": NUM_EPOCHS,
        "lora_rank": LORA_RANK,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_length": MAX_LENGTH,
    }
    json.dump(hyperparameters, open(os.path.join(output_dir, 'hyperparameters.json'), 'w'))

    # Save the model
    model.save_pretrained(output_dir)  # Local saving
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    fine_tune_qwen(
        model_name='Qwen2.5-VL-7B-Instruct',
        image_directory='data/original_images',
        prompt=prompt0,
        comment='sft_0'
    )