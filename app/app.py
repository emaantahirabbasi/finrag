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

st.title("📊 FinRAG")
st.caption("Hallucination-Aware RAG System for Financial Document Q&A")

st.warning(
    "**Live demo notice:** This cloud deployment runs on Streamlit's free tier, "
    "which has strict limits on the Google Gemini API. Full functionality is "
    "demonstrated in the verified results below and can be reproduced locally. "
    "See the [GitHub repo](https://github.com/emaantahirabbasi/finrag) for full code."
)

st.header("About This Project")
st.write(
    "**FinRAG** is a production-grade Retrieval-Augmented Generation (RAG) system "
    "for financial document Q&A, with an explicit hallucination verification layer."
)

st.header("🎯 What It Does")
st.markdown("""
1. **Ingests** SEC 10-K filings from Apple, Microsoft, Tesla, and Amazon (2,064 chunks total)
2. **Retrieves** the most relevant passages using semantic search (ChromaDB)
3. **Generates** cited answers using Google Gemini
4. **Verifies** each claim against retrieved evidence — flags unsupported claims
5. **Abstains** when evidence is weak (instead of hallucinating)
""")

st.header("📊 Verified Results (from local run)")

st.subheader("Example 1 — Direct Fact Retrieval")
st.markdown("**Q:** What was Apple's total net sales in 2024?")
st.markdown("**A:** Apple's total net sales in 2024 was **$391,035 million** [Source 1, Source 2, Source 4].")
st.success("✅ VERDICT: SUPPORTED | Confidence: HIGH")

st.subheader("Example 2 — Cross-Company Comparison")
st.markdown("**Q:** Compare Apple's and Microsoft's revenue.")
st.markdown("**A:** Apple reported $391,035 million vs Microsoft's $245,122 million — Apple higher by $145,913 million.")
st.warning(
    "⚠️ VERDICT: PARTIAL | 1 unsupported claim detected:\n\n"
    "*\"$145,913 million difference\" — this is a derived calculation, not stated in the sources.*\n\n"
    "This is the hallucination detection working as designed — it distinguishes between sourced claims and inferred ones."
)

st.header("🛠 Tech Stack")
st.markdown("""
- **LLM:** Google Gemini 3.8 Flash
- **Embeddings:** gemini-embedding-001
- **Vector DB:** ChromaDB (2,064 chunks)
- **Framework:** LangChain
- **UI:** Streamlit
- **Document Parsing:** PyPDF, BeautifulSoup
""")

st.header("🚀 Run Locally")
st.code("""
git clone https://github.com/emaantahirabbasi/finrag.git
cd finrag
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
python src/ingest.py
python src/query.py
streamlit run app/app.py
""", language="bash")

st.header("📎 Links")
st.markdown("""
- 📂 [GitHub Repository](https://github.com/emaantahirabbasi/finrag)
- 📖 [README with Architecture](https://github.com/emaantahirabbasi/finrag/blob/main/README.md)
""")

st.header("👤 Author")
st.write("Emaan Tahir Abbasi — Data Science Student, Women University of Azad Jammu & Kashmir")

st.divider()
st.caption("FinRAG v1.0 · Built with Streamlit, LangChain, ChromaDB, and Google Gemini")
