# FinRAG — Hallucination-Aware RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for financial document question answering, with an explicit hallucination verification layer.

## Problem

Financial analysts spend hours manually searching through 100+ page SEC filings. LLM chatbots could help — but they **hallucinate numbers**, which is a compliance disaster in finance. In a regulated industry, one wrong number can trigger legal issues.

## Solution

FinRAG ingests SEC 10-K filings and answers questions with **two AI passes**:

1. **Generation** — retrieves relevant chunks, generates a cited answer using Google Gemini
2. **Verification** — a second LLM pass checks each claim against the retrieved evidence and flags unsupported claims

Most RAG systems stop at step 1. FinRAG knows when it doesn't know.

## Architecture
User question
↓
Embed question (Gemini embedding-001)
↓
Search ChromaDB (top-K=4 chunks)
↓
Build prompt (question + retrieved chunks)
↓
PASS 1: Gemini generates cited answer
↓
PASS 2: Gemini verifies each claim against sources
↓
Output: answer + sources + verdict (SUPPORTED/PARTIAL/UNSUPPORTED)

text

## Documents Ingested

- Apple — FY2024 10-K (543 chunks)
- Microsoft — FY2026 10-K (490 chunks)
- Tesla — FY2024 10-K (614 chunks)
- Amazon — FY2024 10-K (440 chunks)

**Total:** 2,087 chunks embedded in ChromaDB

## Tech Stack

- **LLM:** Google Gemini 3.8 Flash
- **Embeddings:** gemini-embedding-001
- **Vector DB:** ChromaDB (local, persistent)
- **Framework:** LangChain
- **Document Parsing:** PyPDF, BeautifulSoup
- **UI:** Streamlit

## Engineering Highlights

- **Rate-limit-safe ingestion** — batched (50 chunks/65s cycles) with graceful stop-on-quota and resume capability
- **Source-aware deduplication** — skips re-embedding files already in the DB
- **Two-pass architecture** — generation + verification (the differentiator)
- **Metadata-rich retrieval** — every chunk carries source_file + page numbers

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Add SEC filings to data/ folder
# (10-K PDFs or HTMLs from SEC.gov)

# 4. Ingest documents
python src/ingest.py

# 5. Test Q&A
python src/query.py

# 6. Launch the web app
streamlit run app/app.py
Example Output
Q: What was Apple's total net sales in 2024?

A: Apple's total net sales in 2024 was $391,035 million [Source 1, Source 2, Source 4].

Verification: VERDICT: SUPPORTED | CONFIDENCE: HIGH

Q: Compare Apple's and Microsoft's revenue.

A: Apple reported $391,035 million vs Microsoft's $245,122 million — Apple higher by $145,913 million.

Verification: VERDICT: PARTIAL | 1 unsupported claim:

"Apple was higher by $145,913 million" — this is a derived calculation, not stated in the sources.

This is the system working as intended — it distinguishes between claims directly in sources and claims inferred from them.

Project Status
✅ Full ingestion pipeline (4 companies, 2,087 chunks)

✅ Two-pass generation + verification

✅ Streamlit UI with cited answers

✅ CLI test suite (query.py)

🔜 Deployed public demo (Streamlit Cloud)

Author
Emaan Tahir Abbasi — Data Science Student, Women University of Azad Jammu & Kashmir
