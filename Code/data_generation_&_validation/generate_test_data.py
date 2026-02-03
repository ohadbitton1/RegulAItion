import json
import random
import re
import os
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prepare_data import load_and_split_pdfs

# 1. Paste your API key here (hidden for security)
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"
# 2. Set the number of chunks to sample (randomly)

# ==========================================
SAMPLE_SIZE1 = 20
SAMPLE_SIZE2 = 7
SAMPLE_SIZE3 = 3
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

openings = ["What", "How", "When", "Is", "Can", "Are", "Does", "Should", "Under", "To", "In", "Who", "Which", "Whose", "For", "Within", "May"]

# Clean prompt
prompt_template1 = """
You are a strict data generator for banking regulation compliance.
Based **STRICTLY** on the provided context, generate **2 distinct** training examples.

Context:
"{context}"

Task:
Generate a JSON LIST containing 1 'Yes' and 1 'No' Question-Answer pairs.

Requirements:
1. **question**: Practical question by a bank employee (e.g., "Am I allowed to...", "Must we report...").
2. **answer**: "Yes" or "No".

Output format: Return ONLY a valid JSON LIST of objects.
"""
# Hard negative prompt
prompt_template2 = """
You are a strict data generator for banking regulation compliance.
Your goal is to generate Negative Training Examples (Hard Negatives) to train the model to recognize when a context DOES NOT contain the answer.

Context:
"{context}"

Task:
Generate a JSON LIST containing 1 distinct 'N.A' (Not Available) training examples.

Requirements for Negative Examples:
1. **question**: Practical question that is TOPICALLY RELATED but UNANSWERABLE based on the context.
    - Example: If the text discusses "Risk Committees", ask about "Risk Committee member salaries" (if not mentioned).
    - Example: Ask about a specific deadline, dollar amount, or job title that is missing from this specific chunk.
2. **answer**: MUST ALWAYS BE "N.A".

Output format: Return ONLY a valid JSON LIST of objects.
"""

# Soft negative prompt
prompt_template3 = """
You are a strict data generator for banking regulation compliance.
Your task is to generate 1 Soft Negative training examples. 

Context:
"{context}"

Task:
Generate a JSON LIST of 1 objects.

Requirements:
1. question: A professional banking question that is TOPICALLY DIFFERENT from the provided context.
    - Use DIVERSE openings for the questions.
    - The question MUST start with the following opening word: "{opening}"
2. answer: MUST always be "N.A".

Output format: Return ONLY a valid JSON LIST of objects.
"""

prompt1 = PromptTemplate(template=prompt_template1, input_variables=["context"])
prompt2 = PromptTemplate(template=prompt_template2, input_variables=["context"])
prompt3 = PromptTemplate(template=prompt_template3, input_variables=["context","opening"])
chain1 = prompt1 | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()
chain3 = prompt3 | llm | StrOutputParser()

def clean_json_string(s):
    match = re.search(r'\[.*\]', s, re.DOTALL)
    if match: return match.group(0)
    return s

def generate_clean():
    print("--- Loading Chunks ---")
    chunks = load_and_split_pdfs()

    # Save Chunks

    if not chunks: 
        print("Error: No chunks found.")
        return

    # === Change: randomly select 100 chunks from all available ===
    # Using min ensures it won't crash if there are fewer than 100 chunks total
    print(f"Selecting {SAMPLE_SIZE1} random chunks from {len(chunks)} available...")
    selected_chunks = random.sample(chunks, min(len(chunks), SAMPLE_SIZE1))

    # ============================================================
    
    results = []
    output_file = "../../Data/final_eval_questions/clean_rag_test.json" # Output File
    
    print(f"\n--- Starting Run (Unified Source Field) ---")
    
    for i, chunk in enumerate(selected_chunks):
        print(f"Processing chunk {i+1}/{len(selected_chunks)}...")
        
        try:
            # Sending to GPT
            response_text = chain1.invoke({"context": chunk.page_content})
            
            cleaned_response = clean_json_string(response_text)
            data_list = json.loads(cleaned_response)
            
            if isinstance(data_list, dict): data_list = [data_list]
            
            # Data from Python (file name and page)
            filename = os.path.basename(chunk.metadata.get('source', 'Unknown'))
            page_num = chunk.metadata.get('page_label', 0) # page number 

            for item in data_list:
                
                # We prepend the document and page at the beginning
                # Final result: 411-10, AML Officer, Section 14(b)
                final_source = f"{filename}, Page {page_num}"

                entry = {
                    "question": item.get("question"),
                    "context": chunk.page_content,
                    "answer": item.get("answer"),
                    "source": final_source
                }
                results.append(entry)
            
            print(f" -> [V] Success! Added {len(data_list)} pairs.")
            
            # Continuous saving
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
        except Exception as e:
            print(f" -> [X] Error on this chunk: {e}")

    print(f"\nDone! Final file: '{output_file}'")


def generate_hard():
    print("--- Loading Chunks ---")
    chunks = load_and_split_pdfs()

    # Save Chunks

    if not chunks: 
        print("Error: No chunks found.")
        return

    # === Change: randomly select 100 chunks from all available ===
    # Using min ensures it won't crash if there are fewer than 100 chunks total
    print(f"Selecting {SAMPLE_SIZE2} random chunks from {len(chunks)} available...")
    selected_chunks = random.sample(chunks, min(len(chunks), SAMPLE_SIZE2))

    # ============================================================
    
    results = []
    output_file = "../../Data/final_eval_questions/hard_rag_test.json" # Output File
    
    print(f"\n--- Starting Run (Unified Source Field) ---")
    
    for i, chunk in enumerate(selected_chunks):
        print(f"Processing chunk {i+1}/{len(selected_chunks)}...")
        
        try:
            # Sending to GPT
            response_text = chain2.invoke({"context": chunk.page_content})
            
            cleaned_response = clean_json_string(response_text)
            data_list = json.loads(cleaned_response)
            
            if isinstance(data_list, dict): data_list = [data_list]
            
            # Data from Python (file name and page)
            filename = os.path.basename(chunk.metadata.get('source', 'Unknown'))
            page_num = chunk.metadata.get('page_label', 0) # page number

            for item in data_list:
                # We prepend the document and page at the beginning
                # Final result: 411-10, AML Officer, Section 14(b)
                final_source = f"{filename}, Page {page_num}"

                entry = {
                    "question": item.get("question"),
                    "context": chunk.page_content,
                    "answer": item.get("answer"),
                    "source": final_source
                }
                results.append(entry)
            
            print(f" -> [V] Success! Added {len(data_list)} pairs.")
            
            # Continuous saving
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
        except Exception as e:
            print(f" -> [X] Error on this chunk: {e}")

    print(f"\nDone! Final file: '{output_file}'")


def generate_soft():
    print("--- Loading Chunks ---")
    chunks = load_and_split_pdfs()

    # Save Chunks

    if not chunks: 
        print("Error: No chunks found.")
        return

    # === Change: randomly select 100 chunks from all available ===
    # Using min ensures it won't crash if there are fewer than 100 chunks total
    print(f"Selecting {SAMPLE_SIZE3} random chunks from {len(chunks)} available...")
    selected_chunks = random.sample(chunks, min(len(chunks), SAMPLE_SIZE3))

    # ============================================================
    
    results = []
    output_file = "../../Data/final_eval_questions/soft_rag_test.json" # Output File
    
    print(f"\n--- Starting Run (Unified Source Field) ---")
    
    for i, chunk in enumerate(selected_chunks):
        print(f"Processing chunk {i+1}/{len(selected_chunks)}...")
        
        opening = random.choice(openings)
        try:
            # Sending to GPT
            response_text = chain3.invoke({"context": chunk.page_content, "opening": opening})
            
            cleaned_response = clean_json_string(response_text)
            data_list = json.loads(cleaned_response)
            
            if isinstance(data_list, dict): data_list = [data_list]
            
            # Data from Python (file name and page)
            filename = os.path.basename(chunk.metadata.get('source', 'Unknown'))
            page_num = chunk.metadata.get('page_label', 0) # page number

            for item in data_list:
                # We prepend the document and page at the beginning
                # Final result: 411-10, AML Officer, Section 14(b)
                final_source = f"{filename}, Page {page_num}"

                entry = {
                    "question": item.get("question"),
                    "context": chunk.page_content,
                    "answer": item.get("answer"),
                    "source": final_source
                }
                results.append(entry)
            
            print(f" -> [V] Success! Added {len(data_list)} pairs.")
            
            # Continuous saving
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
        except Exception as e:
            print(f" -> [X] Error on this chunk: {e}")

    print(f"\nDone! Final file: '{output_file}'")

def mix_dat_tst():
        
    # --------------------------
    # Load the JSON files
    # --------------------------
    with open("../../Data/final_eval_questions/hard_rag_test.json", "r", encoding="utf-8") as f:
        data1 = json.load(f)

    with open("../../Data/final_eval_questions/soft_rag_test.json", "r", encoding="utf-8") as f:
        data2 = json.load(f)

    with open("../../Data/final_eval_questions/clean_rag_test.json", "r", encoding="utf-8") as f:
        data3 = json.load(f)

    # --------------------------
    # Combine and shuffle
    # --------------------------
    combined = data1 + data2 + data3
    random.shuffle(combined)

    # --------------------------
    # Define output folder and file (existing folder)
    # --------------------------
    output_file = "../../Data/final_eval_questions/complete_rag_test.json" # Output File

    # --------------------------
    # Save combined JSON
    # --------------------------
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    print(f"Mixed {len(data1)} + {len(data2)} + {len(data3)} items → {len(combined)} total")

if __name__ == "__main__":
    generate_clean()
    generate_hard()
    generate_soft()
    mix_dat_tst()