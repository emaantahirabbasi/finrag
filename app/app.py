import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = Path("chroma_db")
MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(
    page_title="FinRAG — Financial Document Q&A",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f3a5f; margin-bottom: 0; }
    .subtitle { font-size: 1rem; color: #666; margin-top: 0; margin-bottom: 2rem; }
    .source-box { background-color: #f0f4f8; border-left: 4px solid #1f3a5f; padding: 10px; margin: 5px 0; border-radius: 4px; font-size: 0.85rem; }
    .answer-box { background-color: #f9fbfd; padding: 20px; border-radius: 8px; border: 1px solid #e0e6ed; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 FinRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Financial document Q&A with source citations — powered by RAG</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **FinRAG** answers questions about SEC 10-K filings using Retrieval-Augmented Generation.
    
    - 📚 Documents: Apple, Microsoft, Tesla, Amazon
    - 🔍 Top-K retrieval: 4 chunks per query
    - 🤖 Model: gemini-3.8-flash
    """)
    st.divider()
    st.caption("Built as a Data Science portfolio project")

@st.cache_resource
def load_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )

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

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Source {i} — {source}, page {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

st.subheader("Ask a question about the filings")

example_qs = [
    "What was Apple's total net sales in 2024?",
    "Compare Apple's and Microsoft's revenue.",
    "What is Tesla's revenue?",
]
selected = st.selectbox("Or pick an example:", [""] + example_qs)

user_q = st.text_input("Your question:", value=selected)

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

if ask_btn and user_q:
    try:
        chain, retriever = load_chain()
        with st.spinner("Retrieving relevant passages and generating answer..."):
            answer = chain.invoke(user_q)
            sources = retriever.invoke(user_q)

        st.markdown("### 💬 Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        st.markdown("### 📎 Sources")
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source_file", "?")
            snippet = doc.page_content[:300].replace("\n", " ")
            st.markdown(
                f'<div class="source-box"><b>[Source {i}]</b> {source} — page {page}<br/><i>{snippet}...</i></div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)[:400]}")
        st.info("If this is a 404, the model name may have changed. If 429, quota exhausted.")

elif ask_btn and not user_q:
    st.warning("Please type a question first.")

st.divider()
st.caption("FinRAG v0.1 · Built with Streamlit, LangChain, ChromaDB, and Google Gemini")