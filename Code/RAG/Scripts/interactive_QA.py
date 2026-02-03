import subprocess
import sys

scripts = {
    "1": "Llama_rag_legalBert.py",
    "2": "Llama_rag_MiniLM.py",
    "3": "Saul_rag_legalBert.py",
    "4": "SaulLm_rag_MiniLM.py",
}

print("Choose a script to run:")
print("1 - Llama + LegalBERT")
print("2 - Llama + MiniLM")
print("3 - SaulLM + LegalBERT")
print("4 - SaulLM + MiniLM")

choice = input("Enter your choice: ")

if choice in scripts:
    subprocess.run([sys.executable, scripts[choice]])
else:
    print("Invalid choice")
