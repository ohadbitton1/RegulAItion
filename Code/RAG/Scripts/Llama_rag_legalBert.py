import os
import sys
import json
import re
import chromadb
from chromadb.api.types import Documents, Embeddings
from unsloth import FastLanguageModel
from peft import PeftModel
from google.colab import drive
from langchain_huggingface import HuggingFaceEmbeddings

# ================= CONFIGURATION =================
PROJECT_ROOT = "/content/drive/MyDrive/RegulAItion"
DB_PATH = os.path.join(PROJECT_ROOT, "Data", "RAG_db_legal")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "Models", "Llama3.1_adapter")

# ================= EMBEDDING ADAPTER =================
class LegalBertAdapter:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="nlpaueb/legal-bert-base-uncased",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.embed_documents(input)

    def embed_query(self, input=None, text=None, **kwargs):
        q = input if input is not None else text
        if q is None: raise ValueError("expected 'input' or 'text'.")
        if isinstance(q, list): return self.model.embed_documents(q)
        return self.model.embed_query(q)

# ================= VERIFICATION LOGIC =================
def tokenize(text):
    if not text: return []
    return re.findall(r"\w+", text.lower())

def verify_citation(citation, context_text):
    """
    Checks if the citation provided by the model exists within the retrieved context.
    Uses fuzzy sequence matching (token by token).
    """
    if not citation or citation == "N.A":
        return "N.A"
    
    cit_words = tokenize(citation)
    chunk_words = tokenize(context_text)

    it = iter(chunk_words)
    is_valid = all(w in it for w in cit_words)
    return "✅ VALID" if is_valid else "❌ HALLUCINATION (Not found in source)"

# ================= SETUP & LOAD =================
def setup_environment():
    print("🚀 Initializing Environment...")
    if not os.path.exists("/content/drive"):
        drive.mount("/content/drive")
    
    if not os.path.exists(ADAPTER_PATH):
        alt = ADAPTER_PATH.replace("Llama", "llama")
        if os.path.exists(alt): return DB_PATH, alt
        raise FileNotFoundError(f"❌ Adapter not found at {ADAPTER_PATH}")
    return DB_PATH, ADAPTER_PATH

def load_system(db_path, adapter_path):
    print("⏳ Loading Model & Adapter (this may take 1-2 mins)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)

    print("🧠 Connecting to LegalDB...")
    embedding_function = LegalBertAdapter()
    client = chromadb.PersistentClient(path=db_path)
    cols = client.list_collections()
    if not cols: raise RuntimeError(f"❌ No collections found at: {db_path}")
    
    col = client.get_collection(name=cols[0].name, embedding_function=embedding_function)
    return model, tokenizer, col

# ================= RAG FUNCTIONS =================
def retrieve(col, query, n_results=3):
    res = col.query(query_texts=[query], n_results=n_results)
    context_list = []
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    if docs and docs[0]:
        for i, doc in enumerate(docs[0]):
            meta = metas[0][i] if metas and metas[0] else {}
            context_list.append(f"Source: {meta.get('source', 'UNK')} (Page {meta.get('page_label', '0')})\nText: {doc}")
    
    return "\n\n".join(context_list)

def generate(model, tokenizer, query, context):
    prompt = f"""### Instruction:
You are an expert regulatory compliance assistant.
Your task is to answer the user's question based STRICTLY on the provided context.

Rules:
1. If the answer is explicitly supported by the text, return "verdict": "Yes" or "No".
2. If the answer is NOT found in the context, return "verdict": "N.A".
3. "quote" must be an EXACT copy of the relevant text segment.
4. Return ONLY a valid JSON object with keys: "verdict", "explanation", "quote", "source_details".

### Input:
Context:
{context}

Question:
{query}

### Response:
"""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return text.split("### Response:")[-1].strip()

def safe_extract_json(ans: str):
    try:
        start = ans.find("{")
        end = ans.rfind("}")
        return json.loads(ans[start : end + 1])
    except:
        return None

# ================= MAIN LOOP =================
def main():
    try:
        db_path, adapter_path = setup_environment()
        model, tokenizer, col = load_system(db_path, adapter_path)

        print("\n✅ SYSTEM ONLINE. Type 'exit' to quit.\n")

        while True:
            q = input("❓ Question: ").strip()
            if q.lower() in ["exit", "quit"]: break
            if not q: continue

            # 1. Retrieval
            ctx = retrieve(col, q)
            
            # 2. Generation
            ans = generate(model, tokenizer, q, ctx)
            data = safe_extract_json(ans)

            if not data:
                print(f"⚠️ Failed to parse model output: {ans}")
                continue

            # 3. Verification (Your Logic)
            v_status = verify_citation(data.get("quote"), ctx)

            # 4. Final Output
            print("\n" + "="*60)
            print(f"VERDICT      : {data.get('verdict')}")
            print(f"VERIFICATION : {v_status}")
            print(f"EXPLANATION  : {data.get('explanation')}")
            if data.get("quote") != "N.A":
                print(f"QUOTE        : \"{data.get('quote')}\"")
                print(f"SOURCE       : {data.get('source_details')}")
            print("="*60 + "\n")

    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    main()