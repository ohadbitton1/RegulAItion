import json
import os
import argparse
from unsloth import FastLanguageModel
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference using Fine-Tuned Llama3/SaulLM model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned adapter")
    parser.add_argument("--output_file", type=str, required=True, help="Output filename (e.g., results.json)")
    
    # Optional: Test file path (defaults to standard location)
    default_test_path = os.path.join("..", "..", "Data", "FT_datasets", "test_dataset.json")
    parser.add_argument("--test_file", type=str, default=default_test_path, help="Path to test dataset")
    
    # Optional: Output directory (defaults to current folder based on user request)
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save results (default: current folder)")

    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    """Load model in 4-bit mode for efficient inference."""
    print(f"⏳ Loading model from: {model_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    return model, tokenizer

def main():
    args = parse_args()

    # Validate input file
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"❌ Test file not found: {args.test_file}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    full_output_path = os.path.join(args.output_dir, args.output_file)

    # Load resources
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"🚀 Starting inference on {len(test_data)} examples...")
    
    results = []
    
    # Alpaca Prompt Template (Must match training format)
    alpaca_template = """### Instruction:
{}

### Input:
{}

### Response:
"""

    # Inference Loop
    for item in tqdm(test_data):
        prompt = alpaca_template.format(item["instruction"], item["input"])
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Generate response
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,   # Sufficient for JSON output
            use_cache=True,
            temperature=0.1,      # Low temperature for deterministic results
        )
        
        # Decode output
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract only the model's response (after the prompt)
        try:
            response_only = generated_text.split("### Response:\n")[-1].strip()
        except IndexError:
            response_only = generated_text # Fallback

        # Append result
        results.append({
            "question": item.get("instruction", ""),
            "ground_truth": item["output"],
            "model_prediction": response_only
        })

    # Save results to JSON
    with open(full_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Success! Results saved to: {full_output_path}")

if __name__ == "__main__":
    main()