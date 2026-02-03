import json
import random
import re
import os
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prepare_data import load_and_split_pdfs

# 1. Paste your API key here (hidden for security)
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# 2. Set the number of chunks to sample (randomly)
SAMPLE_SIZE = 518

# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Updated prompt: request a single unified "source" field from the model
prompt_template = """
You are a strict data generator for banking regulation compliance.
Based **STRICTLY** on the provided context, generate **4 distinct** training examples.

Context:
"{context}"

Task:
Generate a JSON LIST containing 2 'Yes' and 2 'No' Question-Answer pairs.

Requirements:
1. **question**: Practical question by a bank employee (e.g., "Am I allowed to...", "Must we report...").
2. **answer**: "Yes" or "No".
3. **citation**: **CRITICAL**: This must be a **VERBATIM COPY-PASTE** from the text.
   - Extract the sentence EXACTLY as it appears in the text, including punctuation and weird spacing.
   - **DO NOT** rephrase, summarize, or fix grammar.
   - **DO NOT** combine two different sentences into one.
   - If the sentence is cut off in the context, copy only what is visible.
4. **explanation**: A short, professional rationale (1 sentence).
   - **STYLE**: Write as a factual statement or principle.
   - **FORBIDDEN**: DO NOT refer to "the text", "the passage", "this section", "the document", or "it says".
   - **BAD EXAMPLE**: "The text mentions that risk assessments are vital."
   - **GOOD EXAMPLE**: "Risk assessments are vital to ensure long-term stability."
5. **source_details**: Format: "Header, Section X" (or just Header).

Output format: Return ONLY a valid JSON LIST of objects.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
chain = prompt | llm | StrOutputParser()

def clean_json_string(s):
    match = re.search(r'\[.*\]', s, re.DOTALL)
    if match: return match.group(0)
    return s

def get_doc_number(filename):
    match = re.match(r'(\d+)', filename)
    if match: return match.group(1)
    return filename

def generate_synthetic_data_banker():
    print("--- Loading Chunks ---")
    chunks = load_and_split_pdfs()

    # Save Chunks

    if not chunks: 
        print("Error: No chunks found.")
        return

    # === Change: randomly select 100 chunks from all available ===
    # Using min ensures it won't crash if there are fewer than 100 chunks total
    print(f"Selecting {SAMPLE_SIZE} random chunks from {len(chunks)} available...")
    selected_chunks = random.sample(chunks, min(len(chunks), SAMPLE_SIZE))

    # ============================================================
    
    results = []
    output_file = "../../Data/raw_datasets/RegulAItion_dataset.json" # Output File
    
    print(f"\n--- Starting Run (Unified Source Field) ---")
    
    for i, chunk in enumerate(selected_chunks):
        print(f"Processing chunk {i+1}/{len(selected_chunks)}...")
        
        try:
            # Sending to GPT
            response_text = chain.invoke({"context": chunk.page_content})
            
            cleaned_response = clean_json_string(response_text)
            data_list = json.loads(cleaned_response)
            
            if isinstance(data_list, dict): data_list = [data_list]
            
            # Data from Python (file name and page)
            filename = os.path.basename(chunk.metadata.get('source', 'Unknown'))
            doc_num = get_doc_number(filename)
            page_num = chunk.metadata.get('page', 0) + 1 # 1 page based

            for item in data_list:
                # The model returns the text (title and section)
                text_location = item.get("source_details", "General")
                
                # We prepend the document and page at the beginning
                # Final result: 411-10, AML Officer, Section 14(b)
                final_source = f"{doc_num}-{page_num}, {text_location}"

                entry = {
                    "question": item.get("question"),
                    "context": chunk.page_content,
                    "answer": item.get("answer"),
                    "citation": item.get("citation"),
                    "explanation": item.get("explanation"),
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

if __name__ == "__main__":
    generate_synthetic_data_banker()