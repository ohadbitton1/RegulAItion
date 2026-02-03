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
SAMPLE_SIZE = 300

# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Updated prompt: request a single unified "source" field from the model
prompt_template = """
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
3. **citation**: CRITICAL: Verbatim copy-paste of the sentence that is MOST SEMANTICALLY RELATED to the question, but still fails to answer it.
    - This trains the model to distinguish between "related text" and "actual answer".
    - DO NOT leave this empty. Find the closest "distractor" sentence.
4. **explanation**: A short rationale explaining exactly what information is missing.
    - STYLE: Write as a factual statement.
    - FORBIDDEN: DO NOT refer to "the text", "the passage", "this section", or "the document".
    - GOOD EXAMPLE: "The regulation does not specify the minimum number of independent directors for the committee."
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
    output_file = "../../Data/raw_datasets/RegulAItion_dataset_hard_neg.json" # Output File
    
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