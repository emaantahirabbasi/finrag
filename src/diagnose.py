from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

vs = Chroma(
    persist_directory="chroma_db",
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
)

data = vs.get()
ids = data.get("ids", [])
metadatas = data.get("metadatas", [])
documents = data.get("documents", [])

print(f"Total chunks in DB: {len(ids)}")
print()

# Count by source
counts = Counter(m.get("source_file", "unknown") for m in metadatas)
print("Chunks per source:")
for src, n in sorted(counts.items()):
    print(f"  {src}: {n}")
print()

# Check ID format — old chunks vs new
print("Sample IDs (first 5):")
for i in range(min(5, len(ids))):
    print(f"  {ids[i]}")
print()

print("Sample IDs (last 5):")
for i in range(max(0, len(ids)-5), len(ids)):
    print(f"  {ids[i]}")
print()

# Check for duplicate content
print("Checking for duplicate content...")
content_hashes = Counter()
for doc in documents:
    import hashlib
    h = hashlib.md5(doc.strip().encode("utf-8")).hexdigest()
    content_hashes[h] += 1

duplicates = {h: n for h, n in content_hashes.items() if n > 1}
print(f"  Unique contents: {len(content_hashes)}")
print(f"  Duplicated contents: {len(duplicates)}")
if duplicates:
    total_dupes = sum(n - 1 for n in duplicates.values())
    print(f"  Total duplicate chunks: {total_dupes}")