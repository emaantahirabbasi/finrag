import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="FinRAG — Financial Document Q&A", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f3a5f; margin-bottom: 0; }
    .subtitle { font-size: 1rem; color: #666; margin-top: 0; margin-bottom: 2rem; }
    .card { background: #f9fbfd; padding: 20px; border-radius: 8px; border: 1px solid #e0e6ed; margin-bottom: 15px; }
    .highlight { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; border-radius: 4px; margin: 15px 0; }
    .result-box { background: #e7f3ff; border-left: 4px solid #1f3a5f; padding: 15px; border-radius: 4px; margin: 10px 0; }
    .verdict-supported { color: #0a7c2f; font-weight: bold; }
    .verdict-partial { color: #d97706; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 FinRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Hallucination-Aware RAG System for Financial Document Q&A</p>', unsafe_allow_html=True)

st.markdown("""
<div class="highlight">
<strong>⚠️ Live demo notice:</strong> This cloud deployment runs on Streamlit's free tier, which has strict limits on the Google Gemini API (used for embeddings + generation). Full functionality is demonstrated in the verified results below and can be reproduced locally by following the instructions. See the <a href="https://github.com/emaantahirabbasi/finrag" target="_blank">GitHub repo</a> for full code and setup.
</div>
""", unsafe_allow_html=True)

st.markdown("## About This Project")
st.info("**FinRAG** is a production-grade Retrieval-Augmented Generation (RAG) system for financial document Q&A, with an explicit hallucination verification layer.")

st.markdown("### 🎯 What It Does")
st.markdown("""
1. **Ingests** SEC 10-K filings from Apple, Microsoft, Tesla, and Amazon (2,064 chunks total)
2. **Retrieves** the most relevant passages using semantic search (ChromaDB)
3. **Generates** cited answers using Google Gemini
4. **Verifies** each claim against retrieved evidence — flags unsupported claims
5. **Abstains** when evidence is weak (instead of hallucinating)
""")

st.markdown("### 📊 Verified Results (from local run)")

st.markdown("""
<div class="result-box">
<strong>Q:</strong> What was Apple's total net sales in 2024?<br><br>
<strong>A:</strong> Apple's total net sales in 2024 was <strong>$391,035 million</strong> [Source 1, Source 2, Source 4].<br><br>
<strong>Verification:</strong> <span class="verdict-supported">✅ VERDICT: SUPPORTED</span> | Confidence: HIGH
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="result-box">
<strong>Q:</strong> Compare Apple's and Microsoft's revenue.<br><br>
<strong>A:</strong> Apple reported $391,035 million vs Microsoft's $245,122 million — Apple higher by $145,913 million.<br><br>
<strong>Verification:</strong> <span class="verdict-partial">⚠️ VERDICT: PARTIAL</span> | 1 unsupported claim detected:<br>
<em>"$145,913 million difference" — this is a derived calculation, not stated in the sources.</em><br><br>
<small>✅ This is the hallucination detection working as designed — it distinguishes between sourced claims and inferred ones.</small>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🛠 Tech Stack")
st.markdown("""
- **LLM:** Google Gemini 3.8 Flash
- **Embeddings:** gemini-embedding-001
- **Vector DB:** ChromaDB (2,064 chunks)
- **Framework:** LangChain
- **UI:** Streamlit
- **Document Parsing:** PyPDF, BeautifulSoup
""")

st.markdown("### 🚀 Run Locally")
st.code("""
git clone https://github.com/emaantahirabbasi/finrag.git
cd finrag
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
python src/ingest.py
python src/query.py
streamlit run app/app.py
""", language="bash")

st.markdown("### 📎 Links")
st.markdown("""
- 📂 [GitHub Repository](https://github.com/emaantahirabbasi/finrag)
- 📖 [README with Architecture](https://github.com/emaantahirabbasi/finrag/blob/main/README.md)
""")

st.markdown("### 👤 Author")
st.write("Emaan Tahir Abbasi — Data Science Student, Women University of Azad Jammu & Kashmir")

st.divider()
st.caption("FinRAG v1.0 · Built with Streamlit, LangChain, ChromaDB, and Google Gemini")
