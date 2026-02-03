import json
import matplotlib.pyplot as plt
import os

# Paths של הקבצים
LLAMA_LEGAL = "../Results/eval_llama_legalbert_Llama3.1_adapter_complete_rag_test_summary.json"
LLAMA_MINI = "../Results/eval_llama_minilm_Llama3.1_adapter_complete_rag_test_summary.json"
SAUL_LEGAL = "../Results/eval_saul_legalbert_saul_adapter_complete_rag_test_summary.json"
SAUL_MINI = "../Results/eval_saul_minilm_saul_adapter_complete_rag_test_summary.json"

EDA_DIR = "../Visuals/Final_EDA"
os.makedirs(EDA_DIR, exist_ok=True)

files = {
    "Llama + LegalBERT": LLAMA_LEGAL,
    "Llama + MiniLM": LLAMA_MINI,
    "SaulLM + LegalBERT": SAUL_LEGAL,
    "SaulLM + MiniLM": SAUL_MINI
}

accuracies = {}
for name, path in files.items():
    abs_path = os.path.abspath(path)
    print(f"Trying to open {name}: {abs_path}")
    
    if os.path.exists(abs_path):
        with open(abs_path, "r") as f:
            data = json.load(f)
            accuracies[name] = data.get("verdict_accuracy", 0)
    else:
        print(f"File not found: {abs_path}")
        accuracies[name] = 0

# יצירת plot
plt.figure(figsize=(12,7))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']  # לדוגמה: כחול, כתום, ירוק, אדום
bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors)


# סיבוב שמות ציר x
plt.xticks(rotation=20)

# הצגת y-axis באחוזים
plt.ylabel("Verdict Accuracy (%)")
plt.ylim(0, 1)
plt.gca().set_yticks([i/10 for i in range(0,11)])
plt.gca().set_yticklabels([f"{int(i*100)}%" for i in plt.gca().get_yticks()])

# הצגת ערכים מעל כל bar בגובה קבוע של 90%
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, 0.97, f"{int(bar.get_height()*100)}%", 
             ha='center', va='bottom', fontweight='bold')


plt.title("Verdict Accuracy Comparison")

# שמירה וסגירה
plt.savefig(os.path.join(EDA_DIR, "verdict_accuracy.png"), dpi=300)
plt.show()
plt.close()









accuracies = {}
for name, path in files.items():
    abs_path = os.path.abspath(path)
    print(f"Trying to open {name}: {abs_path}")
    
    if os.path.exists(abs_path):
        with open(abs_path, "r") as f:
            data = json.load(f)
            accuracies[name] = data.get("hallucination_rate_total", 0)
    else:
        print(f"File not found: {abs_path}")
        accuracies[name] = 0

# יצירת plot
plt.figure(figsize=(12,7))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']  # לדוגמה: כחול, כתום, ירוק, אדום
bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors)


# סיבוב שמות ציר x
plt.xticks(rotation=20)

# הצגת y-axis באחוזים
plt.ylabel("Hallucination Rate (%)")
plt.ylim(0, 1)
plt.gca().set_yticks([i/10 for i in range(0,11)])
plt.gca().set_yticklabels([f"{int(i*100)}%" for i in plt.gca().get_yticks()])

# הצגת ערכים מעל כל bar בגובה קבוע של 90%
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, 0.97, f"{int(bar.get_height()*100)}%", 
             ha='center', va='bottom', fontweight='bold')


plt.title("Hallucination Rate Comparison")

# שמירה וסגירה
plt.savefig(os.path.join(EDA_DIR, "hallucination_rate.png"), dpi=300)
plt.show()
plt.close()
