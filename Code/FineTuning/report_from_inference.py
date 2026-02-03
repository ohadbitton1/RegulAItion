import json
import argparse
import os

# Define the default directory for results relative to the 'Code' folder
RESULTS_DIR = os.path.join("..", "..", "Results")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate KPI Report from Inference JSON Results")
    
    # Input is just the filename now, not the full path
    parser.add_argument("--file", type=str, required=True, help="Filename inside the Results folder (e.g., 'saul_results.json')")
    
    return parser.parse_args()

def normalize_answer(text):
    """
    Normalizes the answer text for fair comparison.
    Converts 'Yes.' -> 'yes', ' N.A ' -> 'n.a', etc.
    """
    if not text:
        return ""
    # Strip whitespace, convert to lowercase, remove trailing dots
    return text.strip().lower().replace('.', '')

def main():
    args = parse_args()
    
    # Construct the full path automatically
    full_path = os.path.join(RESULTS_DIR, args.file)
    
    # Validate file existence
    if not os.path.exists(full_path):
        print(f"❌ Error: File not found at {full_path}")
        print(f"   (Make sure the file is in the '{RESULTS_DIR}' folder)")
        return

    # Load results
    print(f"📂 Loading: {full_path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    total_samples = len(results)
    print(f"📊 Evaluating {total_samples} samples...\n")

    # Initialize counters
    valid_json_count = 0        # Count of syntactically valid JSONs
    correct_overall_count = 0   # Count of correct answers (Yes/No/N.A)
    
    na_ground_truth_count = 0   # Total 'N.A' cases in Ground Truth
    na_correct_prediction_count = 0 # Correct 'N.A' predictions by model

    for item in results:
        prediction_str = item.get("model_prediction", "")
        ground_truth_str = item.get("ground_truth", "")

        # --- KPI 1: JSON Format Validity ---
        pred_json = None
        gt_json = None
        
        try:
            # Attempt to parse model output as JSON
            pred_json = json.loads(prediction_str)
            valid_json_count += 1
        except json.JSONDecodeError:
            # If JSON is broken, skip content evaluation
            continue

        # Parse Ground Truth
        try:
            gt_json = json.loads(ground_truth_str)
        except:
            continue

        # Extract and normalize 'answer' fields
        pred_ans = normalize_answer(pred_json.get("answer", ""))
        gt_ans = normalize_answer(gt_json.get("answer", ""))

        # --- KPI 2: Overall Accuracy ---
        if pred_ans == gt_ans:
            correct_overall_count += 1

        # --- KPI 3: N.A Accuracy (Hallucination Check) ---
        if gt_ans == "na" or gt_ans == "n.a": 
            na_ground_truth_count += 1
            if pred_ans == "na" or pred_ans == "n.a":
                na_correct_prediction_count += 1

    # ================= CALCULATE SCORES =================
    
    # KPI 1: JSON Format Rate
    json_score = (valid_json_count / total_samples) * 100
    
    # KPI 2: Overall Accuracy
    accuracy_score = (correct_overall_count / total_samples) * 100
    
    # KPI 3: N.A Accuracy
    if na_ground_truth_count > 0:
        na_score = (na_correct_prediction_count / na_ground_truth_count) * 100
    else:
        na_score = 0.0

    # ================= PRINT REPORT =================
    print("-" * 50)
    print(f"✅ KPI 1 - Valid JSON Format:   {json_score:.2f}%  ({valid_json_count}/{total_samples})")
    print(f"🎯 KPI 2 - Overall Accuracy:    {accuracy_score:.2f}%  ({correct_overall_count}/{total_samples})")
    print(f"🛡️  KPI 3 - 'N.A' Accuracy:      {na_score:.2f}%  ({na_correct_prediction_count}/{na_ground_truth_count})")
    print("-" * 50)

    # Final Verdict
    if json_score < 98:
        print("⚠️  CRITICAL: JSON format is unstable.")
    elif na_score < 85:
        print("⚠️  WARNING: High hallucination risk (low N.A accuracy).")
    else:
        print("🚀 EXCELLENT RESULTS! Model is production-ready.")

if __name__ == "__main__":
    main()