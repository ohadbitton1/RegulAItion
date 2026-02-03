import os
import json
import re
import chromadb
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from google.colab import drive
from langchain_huggingface import HuggingFaceEmbeddings

# ================= CONFIGURATION =================
PROJECT_ROOT = "/content/drive/MyDrive/RegulAItion"

# ✅ LegalBERT vector DB (already embedded)
DB_PATH = os.path.join(PROJECT_ROOT, "Data", "RAG_db_legal")

# ✅ Saul adapter
SAUL_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "Models", "saul_adapter")

# ✅ Saul base model (from your logs)
SAUL_BASE_MODEL = "Equall/Saul-7B-Instruct-v1"

MAX_SEQ_LEN = 2048
TOP_K = 3

# ✅ LegalBERT embedder for QUERY ONLY (DB already has vectors)
LEGALBERT_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


# ================= SETUP =================
def setup_environment():
    print("🚀 Initializing Environment...")
    if not os.path.exists("/content/drive"):
        drive.mount("/content/drive")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ Legal DB not found at: {DB_PATH}")

    if not os.path.exists(SAUL_ADAPTER_PATH):
        raise FileNotFoundError(f"❌ Saul adapter not found at: {SAUL_ADAPTER_PATH}")

    return DB_PATH, SAUL_ADAPTER_PATH


def build_query_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=LEGALBERT_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def format_source(meta: dict) -> str:
    # metadata["source"] looks like "..\\..\\Data\\Regulatory_Rules\\310_et.pdf"
    raw = meta.get("source", "UNK")
    raw = raw.replace("\\", "/")
    file_name = os.path.basename(raw) if raw else "UNK"

    # Prefer page_label (string like "33"), fallback to page (0-based) + 1
    page_label = meta.get("page_label", None)
    if page_label is None or str(page_label).strip() == "":
        page = meta.get("page", None)
        if isinstance(page, int):
            page_label = str(page + 1)
        else:
            page_label = "0"

    return f"{file_name}, Page {page_label}"


# ================= LOAD SYSTEM =================
def load_system(db_path, adapter_path):
    print("⏳ Loading Saul base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SAUL_BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    print("🧩 Loading LoRA adapter (Saul)...")
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)

    print("🧠 Connecting to Chroma DB (RAG_db_legal, persisted vectors)...")
    client = chromadb.PersistentClient(path=db_path)

    cols = client.list_collections()
    if not cols:
        raise RuntimeError(f"❌ No collections found in: {db_path}")

    # IMPORTANT: do NOT pass embedding_function (avoid conflict with persisted config)
    col = client.get_collection(name=cols[0].name)

    # Query embedder only (LegalBERT)
    embedder = build_query_embedder()

    return model, tokenizer, col, embedder


# ================= RETRIEVAL =================
def retrieve(col, embedder, query, n_results=TOP_K):
    # DB already has vectors; we only embed the query with LegalBERT
    q_emb = embedder.embed_query(query)  # list[float]
    res = col.query(query_embeddings=[q_emb], n_results=n_results)

    chunks = []
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    if docs and docs[0]:
        for i, doc in enumerate(docs[0]):
            meta = metas[0][i] if metas and metas[0] else {}
            src = format_source(meta)

            chunks.append(
                f"[CHUNK {i+1}]\n"
                f"SOURCE: {src}\n"
                f"TEXT:\n{doc}"
            )

    return "\n\n".join(chunks), res


# ================= GENERATION =================
def generate(model, tokenizer, query, context):
    prompt = f"""### Instruction:
You are an expert regulatory compliance assistant.
Answer the user's question based STRICTLY on the provided context chunks.

You MUST return ONLY a valid JSON object with EXACT keys:
"verdict", "explanation", "quote", "source"

Rules:
1) If the context explicitly answers the question, set "verdict" to "Yes" or "No".
2) If the answer is NOT found in the context, set "verdict" to "N.A".
3) If verdict is "N.A", set:
   "quote": "N.A"
   "source": "N.A"
4) If verdict is NOT "N.A":
   - "quote" MUST be an EXACT copy-paste substring from one of the TEXT blocks (not rephrased).
   - "source" MUST be an EXACT copy of the corresponding SOURCE line, formatted like:
     "311_et.pdf, Page 33"
5) Never output null/None for quote or source. Use strings only.
6) Return JSON only. No extra text.

### Context:
{context}

### Question:
{query}

### Response:
"""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
    text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return text.split("### Response:")[-1].strip()


def parse_json_from_text(ans: str) -> dict:
    start = ans.find("{")
    end = ans.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(ans[start:end + 1])


def normalize_model_output(data: dict, retrieved_res: dict) -> dict:
    verdict = str(data.get("verdict", "N.A")).strip()
    quote = data.get("quote", "N.A")
    source = data.get("source", "N.A")

    nullish = {None, "", "None", "none", "null", "NULL"}

    if verdict == "N.A":
        return {
            "verdict": "N.A",
            "explanation": str(data.get("explanation", "")).strip(),
            "quote": "N.A",
            "source": "N.A",
        }

    # fallback: if model gave Yes/No but quote/source missing, use top retrieved chunk
    if quote in nullish or source in nullish:
        docs = retrieved_res.get("documents", [])
        metas = retrieved_res.get("metadatas", [])
        if docs and docs[0]:
            fallback_doc = docs[0][0]
            fallback_meta = metas[0][0] if metas and metas[0] else {}
            fallback_source = format_source(fallback_meta)

            fallback_quote = fallback_doc[:350].strip()

            if quote in nullish:
                quote = fallback_quote
            if source in nullish:
                source = fallback_source

    if quote in nullish:
        quote = "N.A"
    if source in nullish:
        source = "N.A"

    # Coerce source to "<pdf>, Page <n>" if model returned a messy path-like string
    if verdict != "N.A" and source != "N.A":
        s = str(source).replace("\\", "/")
        file_name = os.path.basename(s)

        m = re.search(r"Page\s*([0-9]+)", s)
        page = m.group(1) if m else None

        if file_name.lower().endswith(".pdf") and page:
            source = f"{file_name}, Page {page}"

    return {
        "verdict": verdict,
        "explanation": str(data.get("explanation", "")).strip(),
        "quote": str(quote),
        "source": str(source),
    }


# ================= MAIN =================
def main():
    try:
        db_path, adapter_path = setup_environment()
        model, tokenizer, col, embedder = load_system(db_path, adapter_path)

        print("\n✅ SAUL SYSTEM READY (RAG_db_legal / LegalBERT). ASKING QUESTIONS...\n")

        while True:
            q = input("❓ Question (or 'exit'): ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit"]:
                break

            ctx, raw_res = retrieve(col, embedder, q, n_results=TOP_K)
            if not ctx:
                print("❌ DB returned no documents.")
                continue

            ans = generate(model, tokenizer, q, ctx)

            try:
                data = parse_json_from_text(ans)
                data = normalize_model_output(data, raw_res)

                print("-" * 50)
                print(f'Verdict: {data.get("verdict")}')
                print(f'Quote:   {data.get("quote")}')
                print(f'Source:  {data.get("source")}')
                print(f'Explain: {data.get("explanation")}')
                print("-" * 50)

            except Exception as e:
                print(f"⚠️ JSON Error ({e}). Raw output:\n{ans}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
