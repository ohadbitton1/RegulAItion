import json
import random
import os

# --------------------------
# Load the JSON files
# --------------------------
with open("../../Data/RegulAItion_dataset_hard_neg.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open("../../Data/RegulAItion_dataset_soft_neg.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

with open("../../Data/RegulAItion_dataset.json", "r", encoding="utf-8") as f:
    data3 = json.load(f)

# --------------------------
# Combine and shuffle
# --------------------------
combined = data1 + data2 + data3
random.shuffle(combined)

# --------------------------
# Define output folder and file (existing folder)
# --------------------------
output_file = "../../Data/RegulAItion_dataset_complete.json" # Output File

# --------------------------
# Save combined JSON
# --------------------------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=4, ensure_ascii=False)

print(f"Mixed {len(data1)} + {len(data2)} + {len(data3)} items → {len(combined)} total")