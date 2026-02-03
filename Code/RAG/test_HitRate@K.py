import os
import json
import argparse
import pandas as pd
import chromadb
from chromadb.api.types import Documents, Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# ================= PATHS =================
PROJECT_ROOT = "/content/drive/MyDrive/RegulAItion"
# נתיב מעודכן לפי עץ הקבצים ששלחת
TEST_FILE_PATH = os.path.join(PROJECT_ROOT, "Data", "final_eval_questions", "complete_rag_test.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results")

DB_PATHS = {
    'minilm': os.path.join(PROJECT_ROOT, "Data", "RAG_db_MiniLM"),
    'legalbert': os.path.join(PROJECT_ROOT, "Data", "RAG_db_legal")
}

# ================= ADAPTER =================
class LegalBertAdapter:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="nlpaueb/legal-bert-base-uncased",
            model_kwargs={"device": "cpu"}, # עובד גם בלי GPU
            encode_kwargs={"normalize_embeddings": True},
        )
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.embed_documents(input)
    def embed_query(self, input=None, text=None, **kwargs):
        q = input if input is not None else text
        if isinstance(q, list): return self.model.embed_documents(q)
        return self.model.embed_query(q)

# ================= MAIN =================
def normalize(text):
    if not text: return ""
    return str(text).lower().replace('.pdf', '').split(',')[0].strip()

def main():
    # קבלת פרמטרים מהרצה
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True, choices=['minilm', 'legalbert'], help="Choose DB type")
    args = parser.parse_args()
    db_type = args.db

    print(f"\n🚀 Starting Retrieval Benchmark for: [{db_type.upper()}]")
    
    # 1. חיבור ל-DB
    db_path = DB_PATHS[db_type]
    print(f"🔌 Connecting to DB at: {db_path}")
    
    client = chromadb.PersistentClient(path=db_path)
    
    if db_type == 'legalbert':
        col = client.get_collection(
            name=client.list_collections()[0].name,
            embedding_function=LegalBertAdapter()
        )
    else:
        # MiniLM (ברירת מחדל)
        col = client.get_collection(name=client.list_collections()[0].name)

    # 2. טעינת המבחן
    if not os.path.exists(TEST_FILE_PATH):
        raise FileNotFoundError(f"❌ Test file not found at: {TEST_FILE_PATH}")

    with open(TEST_FILE_PATH, 'r') as f:
        test_data = json.load(f)

    hits = 0
    total = 0
    results = []
    
    print(f"📉 Checking {len(test_data)} questions...")
    
    for item in tqdm(test_data):
        query = item.get('question')
        gt_source = item.get('source')
        
        # דילוג על שאלות ללא מקור
        if not gt_source or str(gt_source).lower() == "n.a":
            continue
            
        total += 1
        
        # שליפה
        try:
            res = col.query(query_texts=[query], n_results=5)
        except Exception as e:
            print(f"⚠️ Error querying '{query}': {e}")
            continue
        
        retrieved_files = []
        if res['metadatas'] and res['metadatas'][0]:
            retrieved_files = [m.get('source', 'UNK') for m in res['metadatas'][0]]
            
        # בדיקה
        is_hit = 0
        norm_gt = normalize(gt_source)
        for rf in retrieved_files:
            if norm_gt in normalize(rf):
                is_hit = 1
                break
        
        if is_hit:
            hits += 1
            
        results.append({
            "Question": query,
            "Target Source": gt_source,
            "Retrieved Candidates": retrieved_files,
            "Hit": is_hit
        })

    # 3. סיכום
    hit_rate = (hits / total) * 100 if total > 0 else 0
    
    print("\n" + "="*40)
    print(f"🎯 HIT RATE RESULT: {db_type.upper()}")
    print("="*40)
    print(f"Valid Questions:       {total}")
    print(f"Successful Hits:       {hits}")
    print(f"Hit Rate @3:           {hit_rate:.2f}%")
    print("="*40)
    
    df = pd.DataFrame(results)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    output_path = os.path.join(RESULTS_DIR, f"retrieval_results_{db_type}.csv")
    df.to_csv(output_path, index=False)
    print(f"📝 Full report saved to: {output_path}")

if __name__ == "__main__":
    main()
