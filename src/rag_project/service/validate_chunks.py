import json
from pathlib import Path
from collections import Counter

from transformers import AutoTokenizer

CHUNKS_FILE = Path("data/processed/sample/chunks.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
chunks = data["chunks"]

token_counts = [
    len(tokenizer.encode(chunk["text"], add_special_tokens=False))
    for chunk in chunks
]

empty_chunks = [
    c["chunk_index"] for c in chunks if not c["text"].strip()
]

duplicates = [
    text for text, count in Counter(c["raw_text"] for c in chunks).items()
    if count > 1
]

missing_headings = [
    c["chunk_index"] for c in chunks if not c["headings"]
]

missing_pages = [
    c["chunk_index"] for c in chunks if c["page_no"] is None
]

print("Total chunks:", len(chunks))
print("Empty chunks:", empty_chunks)
print("Duplicate chunks:", len(duplicates))
print("Missing headings:", len(missing_headings))
print("Missing page numbers:", len(missing_pages))
print("Minimum tokens:", min(token_counts))
print("Maximum tokens:", max(token_counts))
print("Average tokens:", round(sum(token_counts) / len(token_counts), 2))
print("Chunks exceeding 500 tokens:", sum(t > 500 for t in token_counts))

print("\nFirst 5 chunks:")
for chunk, tokens in zip(chunks[:5], token_counts[:5]):
    print(f"\nChunk {chunk['chunk_index']} | {tokens} tokens")
    print(chunk["text"][:300])

