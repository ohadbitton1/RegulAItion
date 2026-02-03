import json
import os
import re
import sys

# ================== CONFIG ==================
CHUNKS_FILE = "chunks.json"
MODEL_OUTPUT_FILE = "model_output.json"
# ============================================


def parse_chunk_ref(chunk_ref):
    doc_part, page_part = chunk_ref.split(",")
    doc_name = doc_part.strip()
    page_label = page_part.replace("page", "").strip()
    return doc_name, page_label


def get_chunk_text(chunks, doc_name, page_label):
    for chunk in chunks:
        source_path = chunk["metadata"].get("source", "")
        source_file = os.path.basename(source_path)

        if (
            source_file == doc_name
            and chunk["metadata"].get("page_label") == page_label
        ):
            return chunk["text"]

    return None


def tokenize(text):
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def citation_in_chunk(citation, chunk_text):
    cit_words = tokenize(citation)
    chunk_words = tokenize(chunk_text)

    it = iter(chunk_words)
    for w in cit_words:
        if w not in it:
            return False
    return True


def main():
    # Load chunks
    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ Missing chunks file: {CHUNKS_FILE}")
        sys.exit(1)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = chunks_data["chunks"]

    # Load model output
    if not os.path.exists(MODEL_OUTPUT_FILE):
        print(f"❌ Missing model output file: {MODEL_OUTPUT_FILE}")
        sys.exit(1)

    with open(MODEL_OUTPUT_FILE, "r", encoding="utf-8") as f:
        model_output = json.load(f)

    citation = model_output.get("quote")
    chunk_ref = model_output.get("source")

    if not citation or not chunk_ref:
        print("❌ Model output must contain 'quote' and 'source'")
        sys.exit(1)

    # Verification
    doc_name, page_label = parse_chunk_ref(chunk_ref)
    chunk_text = get_chunk_text(chunks, doc_name, page_label)

    if chunk_text is None:
        result = "CHUNK_NOT_FOUND"
    else:
        result = citation_in_chunk(citation, chunk_text)

    print("=== VERIFICATION RESULT ===")
    print("Chunk ref :", chunk_ref)
    print("Result    :", result)


if __name__ == "__main__":
    main()
