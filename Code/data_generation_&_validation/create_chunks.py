import os
import json
from collections import Counter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def serialize_chunks(chunks):
    return [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]


def load_and_split_pdfs():

    # Path to the data folder
    directory_path = "../../Data/Regulatory_Rules" # Regulation Documents pdf files path
    
    print(f"--- Starting Process ---")
    print(f"Looking for PDFs in: {os.path.abspath(directory_path)}")
    
    # 1. Load all files from the folder
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    
    if not documents:
        print("ERROR: No documents found! Please check the 'data' folder.")
        return []

    print(f"\nSuccessfully loaded {len(documents)} pages from the PDFs.")

    # 2. Set up the Splitter (divide into segments)
    # chunk_size=1500: size that allows including a full regulatory section and context
    # chunk_overlap=200: overlap to prevent losing information between paragraphs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=130,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # 3. Perform the actual splitting
    chunks = text_splitter.split_documents(documents)
    
    print(f"\n------------------------------------------------")
    print(f"Total chunks created: {len(chunks)}")
    print(f"------------------------------------------------")
    
    # 4. Statistical check – how many chunks were generated from each file?
    # This ensures all documents (310, 311, etc.) are included
    print("\n--- Sources Statistics (Chunks per File) ---")
    if chunks:
        # Extract the file name from each chunk
        sources = [os.path.basename(chunk.metadata.get('source', 'Unknown')) for chunk in chunks]
        stats = Counter(sources)
        
        for filename, count in stats.items():
            print(f" [V] {filename}: {count} chunks")
    
    # 5. Preview the first chunk to check correctness
    if chunks:
        print("\n--- Preview of the first chunk ---")
        first_chunk = chunks[0]
        source_name = os.path.basename(first_chunk.metadata.get('source', 'Unknown'))
        print(f"Source: {source_name}")
        print("Content (first 500 chars):")
        print(first_chunk.page_content[:500])
        print("...")
        
    # 6. Serialize chunks for JSON
    serialized_chunks = serialize_chunks(chunks)

    data = {
        "meta": {
            "total_chunks": len(serialized_chunks),
            "chunk_size": 900,
            "chunk_overlap": 130,
            "source_dir": os.path.abspath(directory_path)
        },
        "chunks": serialized_chunks
    }

    # 7. Save to JSON
    output_path = "../../Data/Chunks/Regulatory_Rules_Chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[V] Chunks saved successfully to: {output_path}\n")

    return chunks

    '''
    
    # Load it later
    with open("documents_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]


    # LangChain Documents
    from langchain.schema import Document

        documents = [
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            for chunk in chunks
        ]

    '''

if __name__ == "__main__":
    final_chunks = load_and_split_pdfs()