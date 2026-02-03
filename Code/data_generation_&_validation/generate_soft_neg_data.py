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
SAMPLE_SIZE = 100

# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Opening variations for the question that will be generated.
openings = ["What", "How", "When", "Is", "Can", "Are", "Does", "Should", "Under", "To", "In", "Who", "Which", "Whose", "For", "Within", "May"]

# Updated prompt: request a single unified "source" field from the model
prompt_template = """
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
3. citation: Pick a RANDOM sentence from the context (verbatim) to serve as noise.
4. explanation: A professional statement stating that the regulation regarding the CURRENT TOPIC does not address the QUESTION TOPIC.

    - CRITICAL STYLE RULE: 
        - DO NOT refer to "the provided text", "the context", "the chunk", or "the user".
        - INSTEAD, refer to the content itself: "The guidelines on [Topic A]...", "The regulation regarding [Topic A]...".

    - BAD Example: "The provided text focuses on risk but does not mention fees." (Sounds like a robot talking to a developer).
    - GOOD Example: "The regulation regarding Risk Management duties does not specify guidelines for ATM maintenance fees."
    - GOOD Example: "Information regarding cloud security is not present in the directives concerning Board of Directors appointments."


5. source_details: Format: "Header, Section X".

Output format: Return ONLY a valid JSON LIST of objects.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context","opening"])
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
    output_file = "../../Data/raw_datasets/RegulAItion_dataset_soft_neg.json" # Output File
    
    print(f"\n--- Starting Run (Unified Source Field) ---")
    
    for i, chunk in enumerate(selected_chunks):
        print(f"Processing chunk {i+1}/{len(selected_chunks)}...")
        
        opening = random.choice(openings)
        try:
            # Sending to GPT
            response_text = chain.invoke({"context": chunk.page_content, "opening": opening})
            
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