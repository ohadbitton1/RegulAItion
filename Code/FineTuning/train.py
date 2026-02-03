import argparse
import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ==========================================
# 0. ARGUMENT PARSER
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama3/SaulLM")
    
    # Required: Which model to train
    parser.add_argument("--model_name", type=str, required=True, help="Model ID (e.g., 'unsloth/llama-3-8b...')")
    
    # Required: Where to save the output adapter
    parser.add_argument("--output_dir", type=str, required=True, help="Folder name for saved adapter")
    
    # Optional: Path to training data (Default points to FT_datasets)
    default_data_path = os.path.join("..", "..", "Data", "FT_datasets", "train_dataset.json")
    parser.add_argument("--train_file", type=str, default=default_data_path, help="Path to training JSON")
    
    parser.add_argument("--max_steps", type=int, default=60, help="Training steps")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"\n🚀 Starting Training")
    print(f"Model: {args.model_name}")
    print(f"Data:  {args.train_file}")
    
    # ==========================================
    # 1. LOAD MODEL
    # ==========================================
    MAX_SEQ_LENGTH = 2048
    load_in_4bit = True 

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto-detect
        load_in_4bit = load_in_4bit,
    )

    # Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # ==========================================
    # 2. DATA & FORMATTING
    # ==========================================
    alpaca_template = """### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    EOS_TOKEN = tokenizer.eos_token 

    def formatting_prompts_func(examples):
        texts = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            text = alpaca_template.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    # Load Dataset
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Dataset not found at: {args.train_file}")

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # ==========================================
    # 3. TRAINER SETUP
    # ==========================================
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = args.max_steps, 
            learning_rate = 2e-4, 
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "checkpoints_temp",
        ),
    )

    # ==========================================
    # 4. TRAIN & SAVE
    # ==========================================
    print("⏳ Training...")
    trainer.train()

    print(f"💾 Saving to {args.output_dir}...")
    model.save_pretrained(args.output_dir) 
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Done!")

if __name__ == "__main__":
    main()