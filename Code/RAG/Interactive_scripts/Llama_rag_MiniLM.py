
import os
import sys
import json
import chromadb
from unsloth import FastLanguageModel
from peft import PeftModel
from google.colab import drive

# ================= CONFIGURATION =================
PROJECT_ROOT = "/content/drive/MyDrive/Colab_Work/Regulation"
DB_PATH = os.path.join(PROJECT_ROOT, "Data", "RAG_db_MiniLM")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "Models", "saul_adapter") 

# ================= SETUP =================
def setup_environment():
    print("🚀 Initializing Environment...")
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

    if not os.path.exists(ADAPTER_PATH):
        alt = ADAPTER_PATH.replace("Saul", "saul")
        if os.path.exists(alt): return DB_PATH, alt
        raise FileNotFoundError(f"❌ Adapter not found at {ADAPTER_PATH}")
    
    return DB_PATH, ADAPTER_PATH

# ================= LOAD SYSTEM =================
def load_system(db_path, adapter_path):
    print(f"⏳ Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Equall/Saul-7B-Instruct-v1",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)

    print(f"🧠 Connecting to DB...")
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(name=client.list_collections()[0].name)
    
    return model, tokenizer, col

# ================= RETRIEVAL =================
def retrieve(col, query):
    res = col.query(query_texts=[query], n_results=3)
    context = []
    if res['documents']:
        for i, doc in enumerate(res['documents'][0]):
            meta = res['metadatas'][0][i]
            context.append(f"Source: {meta.get('source', 'UNK')} (Page {meta.get('page_label', '0')})\nText: {doc}")
    return "\n\n".join(context)

# ================= GENERATION (PRO PROMPT) =================
def generate(model, tokenizer, query, context):
    # פרומפט משופר ומקצועי
    prompt = f"""### Instruction:
You are an expert regulatory compliance assistant. 
Your task is to answer the user's question based STRICTLY on the provided context.

Rules:
1. If the answer is explicitly supported by the text, return "verdict": "Yes" or "No".
2. If the answer is NOT found in the context, return "verdict": "N.A".
3. If verdict is "N.A", set "quote" to "N.A" and "source_details" to "N.A".
4. "quote" must be an EXACT copy of the relevant text segment.
5. Return ONLY a valid JSON object with keys: "verdict", "explanation", "quote", "source_details".

### Input:
Context:
{context}

Question:
{query}

### Response:
"""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].split("### Response:")[-1].strip()

# ================= MAIN =================
def main():
    try:
        db_path, adapter_path = setup_environment()
        model, tokenizer, col = load_system(db_path, adapter_path)
        
        print("\n✅ SYSTEM READY. ASKING QUESTIONS (PRO MODE)...\n")
        
        while True:
            q = input("❓ Question (or 'exit'): ")
            if q.lower() in ['exit', 'quit']: break
            
            ctx = retrieve(col, q)
            if not ctx: 
                print("❌ DB returned no documents.")
                continue
                
            ans = generate(model, tokenizer, q, ctx)
            
            # הדפסת דיבאג
            # print(f"\n🐛 DEBUG RAW OUTPUT:\n{ans}\n") 

            try:
                clean_json = ans[ans.find('{'):ans.rfind('}')+1]
                data = json.loads(clean_json)
                
                print("-" * 50)
                print(f"Verdict: {data.get('verdict')}")
                
                # תצוגה חכמה יותר: מציג ציטוט ומקור רק אם יש תשובה
                if data.get('verdict') != 'N.A':
                    print(f"Quote:   {data.get('quote')}")
                    src = data.get('source') or data.get('source_details') or "N/A"
                    print(f"Source:  {src}")
                else:
                    print(f"Quote:   N.A") # נקי יותר
                
                print(f"Explain: {data.get('explanation')}")
                print("-" * 50)
            except:
                print("⚠️ JSON Error. Raw output:", ans)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
