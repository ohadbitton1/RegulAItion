#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid (non-interactive) RAG evaluation runner for a single (model+adapter) + (vector DB/encoder) combo.

Runs over a JSON test set (like Data/final_eval_questions/complete_rag_test.json),
generates model JSON answers (verdict/explanation/quote/source), and writes:
1) detailed CSV with per-question outputs
2) summary JSON with verdict accuracy + hallucination rate

Designed to be compatible with your existing Unsloth+LoRA adapters and ChromaDB persisted vectors.
"""

import os
import json
import argparse
import shutil
from datetime import datetime

import pandas as pd
import torch
import chromadb

from unsloth import FastLanguageModel
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# Defaults inferred from your repo structure
# -----------------------------
DEFAULT_PROJECT_ROOT = "/content/drive/MyDrive/RegulAItion"

DEFAULT_DB_BY_TYPE = {
    "legalbert": ("Data", "RAG_db_legal"),
    "minilm": ("Data", "RAG_db_MiniLM"),   # in your tree listing
}

# Fallback legacy name used by some of your scripts
LEGACY_MINILM_DB = ("Data", "RAG_db_all")

DEFAULT_ADAPTER_BY_MODEL = {
    "llama": ("Models", "Llama3.1_adapter"),
    "saul": ("Models", "saul_adapter"),
}

BASE_MODEL_BY_MODEL = {
    "llama": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "saul": "Equall/Saul-7B-Instruct-v1",
}

EMBEDDER_BY_DB_TYPE = {
    "legalbert": "nlpaueb/legal-bert-base-uncased",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


# -----------------------------
# Utilities
# -----------------------------
def ensure_drive_mounted():
    # Safe in non-Colab too
    if os.path.exists("/content") and (not os.path.exists("/content/drive")):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception:
            # Not running in Colab, ignore
            pass


def resolve_default_path(project_root: str, parts: tuple[str, ...]) -> str:
    return os.path.join(project_root, *parts)


def resolve_db_path(project_root: str, db_type: str, db_path_arg: str | None) -> str:
    if db_path_arg:
        return os.path.abspath(db_path_arg)

    candidate = resolve_default_path(project_root, DEFAULT_DB_BY_TYPE[db_type])
    if os.path.exists(candidate):
        return candidate

    # Legacy fallback for minilm db
    if db_type == "minilm":
        legacy = resolve_default_path(project_root, LEGACY_MINILM_DB)
        if os.path.exists(legacy):
            return legacy

    raise FileNotFoundError(
        f"DB path not found. Tried: {candidate}" + (f" and legacy {legacy}" if db_type == "minilm" else "")
    )


def resolve_adapter_path(project_root: str, model: str, adapter_path_arg: str | None) -> str:
    if adapter_path_arg:
        return os.path.abspath(adapter_path_arg)

    candidate = resolve_default_path(project_root, DEFAULT_ADAPTER_BY_MODEL[model])
    if os.path.exists(candidate):
        return candidate

    # Case-insensitive fallback
    # (e.g., 'Llama' vs 'llama' differences on Windows vs Linux)
    alt = candidate.replace("Llama", "llama")
    if os.path.exists(alt):
        return alt

    raise FileNotFoundError(f"Adapter path not found. Tried: {candidate} (and {alt})")


def maybe_copy_adapter_to_local(adapter_path: str, work_dir: str, enabled: bool) -> str:
    """
    Copies adapter dir from Drive to local disk to avoid "Repo ID" issues and speed up loads.
    Returns path to the adapter to load from.
    """
    if not enabled:
        return adapter_path

    adapter_path = os.path.abspath(adapter_path)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    os.makedirs(work_dir, exist_ok=True)
    local_dst = os.path.join(work_dir, os.path.basename(adapter_path.rstrip("/")))

    # Clean + copy
    if os.path.exists(local_dst):
        shutil.rmtree(local_dst)

    shutil.copytree(adapter_path, local_dst)
    return local_dst


def load_query_embedder(db_type: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDER_BY_DB_TYPE[db_type],
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def format_source(meta: dict) -> str:
    """
    Standardize as: "<pdf>, Page <n>"
    based on your Saul scripts (page_label preferred; else page+1).
    """
    raw = str(meta.get("source", "UNK")).replace("\\", "/")
    file_name = os.path.basename(raw) if raw else "UNK"

    page_label = meta.get("page_label", None)
    if page_label is None or str(page_label).strip() == "":
        page = meta.get("page", None)
        if isinstance(page, int):
            page_label = str(page + 1)
        else:
            page_label = "0"

    return f"{file_name}, Page {page_label}"


def open_collection(client: chromadb.PersistentClient, preferred_name: str | None):
    cols = client.list_collections()
    if not cols:
        raise RuntimeError("No collections found in the provided ChromaDB path.")

    if preferred_name:
        try:
            return client.get_collection(name=preferred_name)
        except Exception:
            pass

    # Fallback to first collection
    return client.get_collection(name=cols[0].name)


def build_context_from_retrieval(res: dict) -> tuple[str, str]:
    """
    Returns:
    - context_str: formatted chunks for the model prompt
    - fallback_source: standardized source of top chunk (for output normalization)
    """
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    chunks = []
    fallback_source = "N.A"
    fallback_quote = "N.A"

    if docs and docs[0]:
        for i, doc in enumerate(docs[0]):
            meta = metas[0][i] if metas and metas[0] else {}
            src = format_source(meta)
            chunks.append(
                f"[CHUNK {i+1}]\n"
                f"SOURCE: {src}\n"
                f"TEXT:\n{doc}"
            )

        # top chunk fallback
        fallback_source = format_source(metas[0][0] if metas and metas[0] else {})
        fallback_quote = (docs[0][0] or "")[:350].strip() if docs[0][0] else "N.A"

    return "\n\n".join(chunks), json.dumps({"source": fallback_source, "quote": fallback_quote})


def make_prompt(context: str, question: str) -> str:
    # Single prompt compatible across all your combos
    return f"""### Instruction:
You are an expert regulatory compliance assistant.
Answer the user's question based STRICTLY on the provided context chunks.

You MUST return ONLY a valid JSON object with EXACT keys:
"verdict", "explanation", "quote", "source"

Rules:
1) If the context explicitly answers the question, set "verdict" to "Yes" or "No".
2) If the answer is NOT found in the context, set "verdict" to "N.A".
3) If verdict is "N.A", set: "quote": "N.A", "source": "N.A"
4) If verdict is NOT "N.A":
   - "quote" MUST be an EXACT copy-paste substring from one of the TEXT blocks (not rephrased).
   - "source" MUST be an EXACT copy of the corresponding SOURCE line, formatted like:
     "311_et.pdf, Page 33"
5) Never output null/None for quote or source. Use strings only.
6) Return JSON only. No extra text.

### Context:
{context}

### Question:
{question}

### Response:
"""


def safe_extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end + 1])


def normalize_verdict(v: str) -> str:
    v = str(v).strip()
    v_up = v.upper().replace(" ", "")
    if v_up in {"YES"}:
        return "Yes"
    if v_up in {"NO"}:
        return "No"
    if v_up in {"N.A", "NA", "N/A", "N.A."}:
        return "N.A"
    return v  # leave as-is for debugging


def normalize_model_output(data: dict, fallback: dict) -> dict:
    """
    Handles:
    - 'source_details' (from your Llama legal script) -> map to 'source'
    - missing quote/source -> fallback to top retrieved chunk
    - verdict == N.A -> force quote/source to N.A
    """
    nullish = {None, "", "None", "none", "null", "NULL"}

    verdict = normalize_verdict(data.get("verdict", "N.A"))
    explanation = str(data.get("explanation", "")).strip()

    # map llama's "source_details" into "source"
    source = data.get("source", data.get("source_details", "N.A"))
    quote = data.get("quote", "N.A")

    if verdict == "N.A":
        return {"verdict": "N.A", "explanation": explanation, "quote": "N.A", "source": "N.A"}

    # fallback if missing
    if quote in nullish or source in nullish:
        if quote in nullish:
            quote = fallback.get("quote", "N.A")
        if source in nullish:
            source = fallback.get("source", "N.A")

    # ensure strings
    if quote in nullish:
        quote = "N.A"
    if source in nullish:
        source = "N.A"

    # enforce "<pdf>, Page <n>" formatting when it looks like a path
    if source != "N.A":
        s = str(source).replace("\\", "/")
        file_name = os.path.basename(s)
        import re
        m = re.search(r"Page\s*([0-9]+)", s)
        page = m.group(1) if m else None
        if file_name.lower().endswith(".pdf") and page:
            source = f"{file_name}, Page {page}"

    return {"verdict": verdict, "explanation": explanation, "quote": str(quote), "source": str(source)}


def load_model_with_adapter(model_key: str, adapter_path: str, max_seq_len: int):
    base_model = BASE_MODEL_BY_MODEL[model_key]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    return base_model, model, tokenizer


def run_one_question(col, embedder, model, tokenizer, question: str, top_k: int, max_new_tokens: int, temperature: float):
    q_emb = embedder.embed_query(question)
    res = col.query(query_embeddings=[q_emb], n_results=top_k)

    context, fallback_json = build_context_from_retrieval(res)
    fallback = json.loads(fallback_json)

    if not context:
        # no docs: force N.A
        return {"verdict": "N.A", "explanation": "DB returned no documents.", "quote": "N.A", "source": "N.A"}, res

    prompt = make_prompt(context, question)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_cache=True,
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    raw = decoded.split("### Response:")[-1].strip()

    data = safe_extract_json(raw)
    norm = normalize_model_output(data, fallback)
    return norm, res


def get_expected_verdict(item: dict) -> str:
    # Your test files use "answer"; some scripts use "expected_verdict"
    v = item.get("expected_verdict", item.get("answer", item.get("expected_answer", "N.A")))
    return normalize_verdict(v)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Hybrid non-interactive RAG eval (model+VDB+testset -> results).")
    parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT, help="Project root folder")
    parser.add_argument("--model", type=str, choices=["llama", "saul"], required=True, help="Which base model to load")
    parser.add_argument("--db_type", type=str, choices=["legalbert", "minilm"], required=True, help="Which embedding/DB type")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test questions JSON (list of items)")
    parser.add_argument("--results_dir", type=str, default=None, help="Results output dir (default: <project_root>/Results)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Override adapter path (dir)")
    parser.add_argument("--db_path", type=str, default=None, help="Override Chroma DB path (dir)")
    parser.add_argument("--collection_name", type=str, default="regulations", help="Preferred Chroma collection name")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved chunks per question")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation length cap")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Model max sequence length")
    parser.add_argument("--copy_adapter", action="store_true", help="Copy adapter to local /content temp before loading")
    parser.add_argument("--adapter_cache_dir", type=str, default="/content/_tmp_adapters", help="Where to copy adapters if enabled")

    args = parser.parse_args()

    ensure_drive_mounted()

    project_root = os.path.abspath(args.project_root)
    results_dir = os.path.abspath(args.results_dir) if args.results_dir else os.path.join(project_root, "Results")

    db_path = resolve_db_path(project_root, args.db_type, args.db_path)
    adapter_path = resolve_adapter_path(project_root, args.model, args.adapter_path)
    adapter_to_load = maybe_copy_adapter_to_local(adapter_path, args.adapter_cache_dir, args.copy_adapter)

    print("Configuration:")
    print(f"  project_root: {project_root}")
    print(f"  model:        {args.model} -> {BASE_MODEL_BY_MODEL[args.model]}")
    print(f"  adapter:      {adapter_path} (loading from: {adapter_to_load})")
    print(f"  db_type:      {args.db_type}")
    print(f"  db_path:      {db_path}")
    print(f"  test_json:    {args.test_json}")
    print(f"  results_dir:  {results_dir}")

    # Load test set
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_items = json.load(f)
    if not isinstance(test_items, list):
        raise ValueError("test_json must contain a JSON list.")

    # Load model + db
    base_model, model, tokenizer = load_model_with_adapter(args.model, adapter_to_load, args.max_seq_len)

    embedder = load_query_embedder(args.db_type)
    client = chromadb.PersistentClient(path=db_path)
    col = open_collection(client, args.collection_name)

    ensure_dir(results_dir)

    # Run eval
    rows = []
    correct = 0
    hallu = 0
    expected_na = 0

    for idx, item in enumerate(test_items, start=1):
        question = item.get("question", "").strip()
        expected = get_expected_verdict(item)

        if expected == "N.A":
            expected_na += 1

        out, _ = run_one_question(
            col=col,
            embedder=embedder,
            model=model,
            tokenizer=tokenizer,
            question=question,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        actual = normalize_verdict(out.get("verdict", "N.A"))

        is_correct = (actual == expected)
        is_hallucination = (expected == "N.A" and actual in {"Yes", "No"})

        correct += int(is_correct)
        hallu += int(is_hallucination)

        rows.append({
            "i": idx,
            "model": args.model,
            "base_model": base_model,
            "adapter": os.path.basename(adapter_path.rstrip("/")),
            "db_type": args.db_type,
            "db_path": db_path,
            "test_set": os.path.basename(args.test_json),
            "question": question,
            "expected_verdict": expected,
            "actual_verdict": actual,
            "is_correct": int(is_correct),
            "is_hallucination": int(is_hallucination),
            "quote": out.get("quote", ""),
            "source": out.get("source", ""),
            "explanation": out.get("explanation", ""),
        })

        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(test_items)}")

    total = len(test_items)
    accuracy = correct / total if total else 0.0
    hallu_rate_total = hallu / total if total else 0.0
    hallu_rate_on_na = hallu / expected_na if expected_na else 0.0

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "base_model": base_model,
        "adapter": os.path.basename(adapter_path.rstrip("/")),
        "db_type": args.db_type,
        "db_path": db_path,
        "test_json": os.path.abspath(args.test_json),
        "total_questions": total,
        "verdict_accuracy": accuracy,
        "correct_count": correct,
        "expected_na_count": expected_na,
        "hallucination_count": hallu,
        "hallucination_rate_total": hallu_rate_total,
        "hallucination_rate_on_expected_na": hallu_rate_on_na,
    }

    # Save
    tag = f"{args.model}_{args.db_type}_{os.path.basename(adapter_path.rstrip('/'))}_{os.path.splitext(os.path.basename(args.test_json))[0]}"
    csv_path = os.path.join(results_dir, f"eval_{tag}.csv")
    json_path = os.path.join(results_dir, f"eval_{tag}_summary.json")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"  Hallucinations: {hallu} (total rate {hallu_rate_total:.3f}, on NA {hallu_rate_on_na:.3f})")


if __name__ == "__main__":
    main()
