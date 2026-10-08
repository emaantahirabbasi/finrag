import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

CHROMA_DIR = Path("chroma_db")
MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(page_title="FinRAG", page_icon="📊", layout="wide")

st.title("📊 FinRAG")
st.caption("Hallucination-Aware RAG System for Financial Document Q&A")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **FinRAG** answers questions about SEC 10-K filings using RAG.
    
    - 📚 Documents: Apple, Microsoft, Tesla, Amazon
    - 🔍 Top-K retrieval: 4 chunks
    - 🤖 Model: gemini-3.8-flash
    """)
    st.divider()
    st.caption("Built as a Data Science portfolio project")

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
1. If the answer is not in the sources, say exactly: "I don't have enough information to answer this."
2. Always cite the sources you use in the format [Source N].
3. Be precise with numbers — copy them exactly as they appear.
4. Do not speculate or use outside knowledge.

Sources:
{context}

Question: {question}

Answer:"""

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
        
        with st.spinner("Retrieving relevant passages and generating answer..."):
            docs = retriever.invoke(user_q)
            
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source_file", "unknown")
                page = doc.metadata.get("page", "?")
                context_parts.append(f"[Source {i} — {source}, page {page}]\n{doc.page_content}")
            context = "\n\n".join(context_parts)
            
            prompt = PROMPT_TEMPLATE.format(context=context, question=user_q)
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        
        # Detect "I don't know" answers
        no_info_phrases = [
            "i don't have enough information",
            "i don't have enough info",
            "i don't have information",
            "not enough information",
            "insufficient information",
            "cannot answer",
            "can't answer",
            "unable to answer",
        ]
        answer_lower = answer.lower()
        is_no_info = any(phrase in answer_lower for phrase in no_info_phrases)
        
        st.markdown("### 💬 Answer")
        st.info(answer)
        
        if not is_no_info:
            st.markdown("### 📎 Sources")
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "?")
                source = doc.metadata.get("source_file", "?")
                snippet = doc.page_content[:300].replace("\n", " ")
                with st.expander(f"[Source {i}] {source} — page {page}"):
                    st.write(snippet + "...")
        else:
            st.caption("🔍 No relevant sources found in the ingested documents.")
    
    except Exception as e:
        st.error(f"⚠️ Error: {type(e).__name__}: {str(e)[:500]}")

elif not user_q:
    st.caption("Type a question above or pick an example.")

st.divider()
st.caption("FinRAG v1.0 · Built with Streamlit, LangChain, ChromaDB, and Google Gemini")
