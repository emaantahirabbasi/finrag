import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

# ---------- Read from Streamlit secrets if available ----------
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

CHROMA_DIR = Path("chroma_db")
MODEL_NAME = "gemini-3.8-flash"
CHROMA_ZIP_URL = "https://github.com/emaantahirabbasi/finrag/releases/download/v1.0-db/chroma_db.zip"

# ---------- Download chroma_db if missing ----------
def ensure_chroma_db():
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return
    import urllib.request, zipfile, tempfile
    st.info("⏳ Downloading vector database...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        urllib.request.urlretrieve(CHROMA_ZIP_URL, tmp.name)
        zip_path = tmp.name
    st.info("📦 Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")
    os.remove(zip_path)

# ---------- Page setup ----------
st.set_page_config(page_title="FinRAG", page_icon="📊", layout="wide")

st.markdown('<h1 style="color:#1f3a5f;margin-bottom:0;">📊 FinRAG</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#666;margin-top:0;">Financial document Q&A with source citations — powered by RAG</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **FinRAG** answers questions about SEC 10-K filings using RAG.
    
    - 📚 Documents: Apple, Microsoft, Tesla, Amazon
    - 🔍 Top-K retrieval: 4 chunks
    - 🤖 Model: gemini-3.8-flash
    """)

# ---------- Load everything once ----------
@st.cache_resource
def load_resources():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY missing")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0
    )
    return retriever, llm

PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the user's question using ONLY the provided sources.

Rules:
1. If the answer is not in the sources, say "I don't have enough information to answer this."
2. Always cite the sources you use in the format [Source N].
3. Be precise with numbers — copy them exactly as they appear.
4. Do not speculate or use outside knowledge.

Sources:
{context}

Question: {question}

Answer:"""

# ---------- Ensure DB ----------
ensure_chroma_db()

# ---------- UI ----------
st.subheader("Ask a question about the filings")

example_qs = [
    "What was Apple's total net sales in 2024?",
    "Compare Apple's and Microsoft's revenue.",
    "What is Tesla's revenue?",
]
selected = st.selectbox("Or pick an example:", [""] + example_qs)
user_q = st.text_input("Your question:", value=selected)

if st.button("🔍 Ask", type="primary") and user_q:
    try:
        retriever, llm = load_resources()
        
        with st.spinner("Retrieving relevant passages..."):
            docs = retriever.invoke(user_q)
        
        # Build context manually
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[Source {i} — {source}, page {page}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        
        prompt = PROMPT_TEMPLATE.format(context=context, question=user_q)
        
        with st.spinner("Generating answer with Gemini..."):
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        
        st.markdown("### 💬 Answer")
        st.markdown(f'<div style="background:#f9fbfd;padding:20px;border-radius:8px;border:1px solid #e0e6ed;">{answer}</div>', unsafe_allow_html=True)
        
        st.markdown("### 📎 Sources")
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source_file", "?")
            snippet = doc.page_content[:300].replace("\n", " ")
            st.markdown(
                f'<div style="background:#f0f4f8;border-left:4px solid #1f3a5f;padding:10px;margin:5px 0;border-radius:4px;font-size:0.85rem;"><b>[Source {i}]</b> {source} — page {page}<br/><i>{snippet}...</i></div>',
                unsafe_allow_html=True
            )
    
    except Exception as e:
        st.error(f"⚠️ Error: {type(e).__name__}: {str(e)[:800]}")

elif not user_q:
    st.caption("Type a question above or pick an example.")

st.divider()
st.caption("FinRAG v0.1 · Built with Streamlit, LangChain, ChromaDB, and Google Gemini")
