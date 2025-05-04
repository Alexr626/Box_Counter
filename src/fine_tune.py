import json
import os
from PIL import Image
from tqdm import tqdm
import wandb
from src.dataset import convert_to_conversation, create_hf_dataset, format_messages_and_images, create_lazy_dataset
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from .prompts import *
from src.vision_utils import get_padding_tokens_ids
LORA_RANK = 16
NUM_EPOCHS = 1
LR = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 8192

def fine_tune_unsloth(model_name: str, image_directory: str, prompt: str, comment: str):
    # Load the model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = False,
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        max_seq_length = MAX_LENGTH,
    )
    # print(tokenizer.image_processor)
    # print(tokenizer.tokenizer)
    # tokenizer.tokenizer.padding_side = "left"
    # print(model.config.vision_config.image_size)
    # print(f"max_length: {tokenizer.tokenizer.model_max_length}")
    model_name = model_name.split('/')[-1]

    output_dir = "models/" + model_name + "_" + comment

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = LORA_RANK,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = LORA_RANK,  # Recommended alpha == r at least
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
    
    print("Creating dataset...")
    # dataset = []
    # for sample in tqdm(train_ds, desc="Creating dataset", unit="sample"):
    #     dataset.append(format_messages_and_images(sample, prompt))
    dataset = create_lazy_dataset(train_ds, prompt)
    # print(dataset[0])
    print("Dataset created.")
    
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
        train_dataset = dataset,
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


def fine_tune_hf(model_name: str, image_directory: str, prompt: str, comment: str):
    # Load model and tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoProcessor.from_pretrained(model_name)
    model_name = model_name.split('/')[-1]

    output_dir = "models/" + model_name + "_" + comment

    # Prepare model for PEFT (QLoRA)
    def find_all_linear_names(target_model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ['multi_modal_projector', 'vision_model']
        for name, module in target_model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset
    ds = create_hf_dataset(image_directory)
    train_ds = ds['train']

    def collate_fn(examples):
        # Get the texts and images
        conversations = [
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': prompt}
                    ]
                },
                {
                    'role': 'assistant',
                    'content': [
                        {"type": "text", "text": str(example['box_count_estimate'])},
                    ]
                }
            ]
            for example in examples
        ]
        texts = [tokenizer.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
        images = [Image.open(example['image']) for example in examples]

        # Tokenize the texts and process the images
        batch = tokenizer(images=images, text=texts, return_tensors="pt", padding=True)
        
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        padding_token_ids = get_padding_tokens_ids(tokenizer)
        labels[torch.isin(labels, padding_token_ids)] = -100
        batch["labels"] = labels

        return batch

    # Initialize wandb
    wandb.init(
        project='Box_counter',
        job_type="training",
        anonymous="allow",
        name=model_name + "_" + comment,
    )

    dataset = create_lazy_dataset(train_ds, prompt)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = collate_fn,
        train_dataset = dataset,
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

    # Train
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save hyperparameters
    hyperparameters = {
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_length": MAX_LENGTH,
    }
    json.dump(hyperparameters, open(os.path.join(output_dir, 'hyperparameters.json'), 'w'))

    wandb.finish()


if __name__ == '__main__':
    # fine_tune_unsloth(
    #     model_name='unsloth/llava-v1.6-mistral-7b-hf',
    #     image_directory='data/original_images',
    #     prompt=prompt0,
    #     comment='sft_0'
    # )
    # fine_tune_unsloth(
    #     model_name="unsloth/Pixtral-12B-2409",
    #     image_directory='data/original_images',
    #     prompt=prompt0,
    #     comment='sft_0'
    # )
    fine_tune_unsloth(
        model_name="unsloth/Pixtral-12B-2409",
        image_directory="data/original_images",
        prompt=prompt3,
        comment='sft_3'
    )