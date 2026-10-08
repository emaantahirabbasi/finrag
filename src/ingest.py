import os
import time
import hashlib
from pathlib import Path
from collections import Counter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

def load_pdf(pdf_path):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = pdf_path.name
    return docs

def load_html(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    doc = Document(
        page_content=clean_text,
        metadata={"source_file": html_path.name}
    )
    return [doc]

def load_documents():
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    html_files = list(DATA_DIR.glob("*.htm")) + list(DATA_DIR.glob("*.html"))

    for pdf_path in pdf_files:
        print(f"📄 Loading PDF: {pdf_path.name}")
        try:
            docs = load_pdf(pdf_path)
            documents.extend(docs)
            print(f"   ✅ {len(docs)} pages loaded")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    for html_path in html_files:
        print(f"🌐 Loading HTML: {html_path.name}")
        try:
            docs = load_html(html_path)
            documents.extend(docs)
            chars = len(docs[0].page_content) if docs else 0
            print(f"   ✅ 1 doc loaded ({chars:,} chars)")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")

    counts = Counter(c.metadata.get("source_file", "unknown") for c in chunks)
    print("   Chunks per source:")
    for src, n in sorted(counts.items()):
        print(f"     - {src}: {n}")

    return chunks

def get_sources_in_db(vectorstore):
    """Return a set of source_file names already in the DB."""
    try:
        data = vectorstore.get()
        metadatas = data.get("metadatas", [])
        sources = {m.get("source_file") for m in metadatas if m}
        return sources
    except Exception:
        return set()

def create_or_load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

def embed_missing_chunks(chunks, vectorstore):
    """Embed only chunks from files NOT already in the DB."""
    existing_sources = get_sources_in_db(vectorstore)
    print(f"\n📊 Files already in DB: {sorted(existing_sources)}")

    # Filter: only chunks from files not yet ingested
    new_chunks = []
    skipped_sources = set()
    for c in chunks:
        src = c.metadata.get("source_file", "unknown")
        if src in existing_sources:
            skipped_sources.add(src)
        else:
            new_chunks.append(c)

    if skipped_sources:
        print(f"⏭️  Skipping (already in DB): {sorted(skipped_sources)}")

    print(f"🔢 New chunks to embed: {len(new_chunks)}")
    if not new_chunks:
        print("✅ Nothing to embed — DB is already complete.")
        return

    batch_size = 50
    total = len(new_chunks)
    failed_batches = []

    for i in range(0, total, batch_size):
        batch = new_chunks[i:i+batch_size]
        # Use deterministic IDs based on source + index
        ids = [
            f"{c.metadata.get('source_file', 'unk')}__{i+j}__{hashlib.md5(c.page_content.encode()).hexdigest()[:8]}"
            for j, c in enumerate(batch)
        ]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        try:
            vectorstore.add_documents(batch, ids=ids)
            print(f"✅ Batch {batch_num}/{total_batches} done ({i+len(batch)}/{total})")
        except Exception as e:
            err = str(e)[:120]
            print(f"❌ Batch {batch_num} failed: {err}")
            failed_batches.append((batch_num, err))
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                print("🛑 Quota exhausted. Stopping gracefully.")
                break

        if i + batch_size < total:
            time.sleep(65)

    if failed_batches:
        print(f"\n⚠️  {len(failed_batches)} batches failed.")

def main():
    print("=" * 50)
    print("🚀 FinRAG — Ingestion (source-aware resume)")
    print("=" * 50)

    documents = load_documents()
    if not documents:
        return

    chunks = split_documents(documents)
    vectorstore = create_or_load_vector_store()
    embed_missing_chunks(chunks, vectorstore)

    print("\n📊 Final DB summary:")
    data = vectorstore.get()
    counts = Counter(m.get("source_file", "unknown") for m in data["metadatas"])
    for src, n in sorted(counts.items()):
        print(f"   {src}: {n}")
    print(f"   Total: {sum(counts.values())}")

    print("\n✅ Ingestion pipeline finished.")

if __name__ == "__main__":
    main()