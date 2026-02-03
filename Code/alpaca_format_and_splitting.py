import json
import os
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
# Paths relative to the 'Code' folder
BASE_DATA_DIR = os.path.join("..", "Data")
INPUT_FILE = os.path.join(BASE_DATA_DIR, "raw_datasets", "RegulAItion_dataset_complete.json")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "FT_datasets")

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_dataset.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_dataset.json")

TEST_SIZE = 0.1  # 10% for testing
# =================================================

def create_alpaca_entry(item):
    """Converts raw data to Alpaca format for RAG."""
    
    system_prompt = "You are a banking regulation expert. Use the provided context to answer the question in JSON format. If the answer is not found, return 'answer': 'N.A'."
    
    instruction = f"{system_prompt}\n\nQuestion:\n{item['question']}"
    input_context = item['context']
    
    # Construct target JSON
    output_json = {
        "answer": item['answer'],
        "citation": item['citation'],
        "explanation": item['explanation'],
        "source_details": item['source']
    }
    output_str = json.dumps(output_json, indent=4, ensure_ascii=False)

    return {
        "instruction": instruction,
        "input": input_context,
        "output": output_str
    }

def main():
    print(f"--- Processing Data ---")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} raw examples.")

    # 2. Convert Format
    alpaca_data = [create_alpaca_entry(item) for item in raw_data]

    # 3. Split Data
    train_data, test_data = train_test_split(alpaca_data, test_size=TEST_SIZE, random_state=42)

    # 4. Save to FT_datasets
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure folder exists

    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Created Train Set: {len(train_data)} items -> {TRAIN_FILE}")
    print(f"✅ Created Test Set:  {len(test_data)} items -> {TEST_FILE}")

if __name__ == "__main__":
    main()