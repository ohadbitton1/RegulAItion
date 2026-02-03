import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
#              CONFIGURATIONS
# ==========================================

# 1. Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define absolute paths relative to this script
# Save DB in the same folder as this script
DB_PATH = "../../Data/RAG_db_MiniLM"

COLLECTION_NAME = "regulations"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

def load_data_from_json():
    """
    Attempts to locate the JSON file based on folder structure.
    """
    # Strategy: Script is in RAG -> Go up to RegulAItion -> Data -> Chunks
    expected_path = os.path.join(BASE_DIR, "../..", "Data", "Chunks", "Regulatory_Rules_Chunks.json")
    # Normalize path for OS
    expected_path = os.path.normpath(expected_path)

    print(f"🔍 Looking for JSON at: {expected_path}")

    if not os.path.exists(expected_path):
        # Backup: Check local folder
        local_path = os.path.join(BASE_DIR, "Regulatory_Rules_Chunks.json")
        if os.path.exists(local_path):
            expected_path = local_path
            print(f"⚠️ Found file locally at: {local_path}")
        else:
            raise FileNotFoundError(f"\n❌ CRITICAL ERROR: Could not find the file at:\n{expected_path}\n\nPlease check that 'Regulatory_Rules_Chunks.json' exists in the 'Data/Chunks' folder.")

    # Load file
    print("✅ File found! Loading data...")
    with open(expected_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    # Handle dict vs list structure
    chunks_list = data.get('chunks', []) if isinstance(data, dict) else data

    for i, chunk in enumerate(chunks_list):
        # Handle different text field names
        text_content = chunk.get('text', chunk.get('content', ''))
        
        # Handle metadata
        meta = chunk.get('metadata', {})
        
        # Clean source path
        source_path = meta.get('source', '')
        filename = os.path.basename(source_path)
        
        clean_meta = {
            "source": filename,
            "page_label": str(meta.get("page_label", "N/A")),
            "title": meta.get("title", "Unknown")
        }

        chunk_id = f"id_{i}"

        # Skip empty chunks
        if text_content: 
            documents.append(text_content)
            metadatas.append(clean_meta)
            ids.append(chunk_id)

    return documents, metadatas, ids

def build_vector_database():
    print("\n--- Starting RAG Database Build (Smart Path Version) ---")
    print(f"Script Location: {BASE_DIR}")

    print(f"Loading model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to DB at: {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH, )
    
    # Reset collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted old collection.")
    except:
        pass 
    
    collection = client.create_collection(name=COLLECTION_NAME)

    # Load data
    try:
        documents, metadatas, ids = load_data_from_json()
    except Exception as e:
        print(e)
        return

    print(f"Loaded {len(documents)} chunks.")
    print("Embedding and saving to ChromaDB...")
    
    # Save in batches
    batch_size = 100 
    total_chunks = len(documents)

    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        collection.add(
            documents=documents[i:end_idx],
            embeddings=model.encode(documents[i:end_idx]).tolist(),
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"   Processed {end_idx}/{total_chunks}...")

    print(f"✅ Success! Database built with {collection.count()} items.")

if __name__ == "__main__":
    build_vector_database()