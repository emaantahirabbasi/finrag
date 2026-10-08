import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from verify import verify_answer, parse_verdict

load_dotenv()

CHROMA_DIR = Path("chroma_db")
MODEL_NAME = "gemini-3.8-flash"

def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        formatted.append(f"[Source {i} — {source}, page {page}]\n{content}")
    return "\n\n".join(formatted)

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

def build_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def ask(question, chain, retriever, run_verification=True):
    print(f"\n❓ {question}")
    print("-" * 60)
    try:
        answer = chain.invoke(question)
        print(f"💬 {answer}")
    except Exception as e:
        print(f"⚠️  Generation failed: {str(e)[:300]}")
        return None, None

    sources = retriever.invoke(question)
    print(f"\n📎 Sources retrieved ({len(sources)}):")
    for i, doc in enumerate(sources, 1):
        page = doc.metadata.get("page", "?")
        print(f"   [{i}] {doc.metadata.get('source_file', '?')} — page {page}")

    if run_verification:
        print(f"\n🛡️  Running hallucination check...")
        try:
            verdict_text = verify_answer(answer, sources)
            parsed = parse_verdict(verdict_text)
            print(f"   VERDICT: {parsed['verdict']}")
            print(f"   CONFIDENCE: {parsed['confidence']}")
            if parsed["supported"]:
                print(f"   ✅ Supported claims: {len(parsed['supported'])}")
            if parsed["unsupported"]:
                print(f"   ⚠️  Unsupported claims: {len(parsed['unsupported'])}")
                for claim in parsed["unsupported"]:
                    print(f"      - {claim}")
        except Exception as e:
            print(f"   ⚠️  Verification failed: {str(e)[:200]}")

    return answer, sources

def main():
    print("=" * 60)
    print("🤖 FinRAG — Financial Document Q&A")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"DB: {CHROMA_DIR}")
    print("=" * 60)

    chain, retriever = build_chain()

    test_questions = [
        "What was Apple's total net sales in 2024?",
        "Compare Apple's and Microsoft's revenue.",
        "What is Tesla's revenue?",
    ]

    for q in test_questions:
        ask(q, chain, retriever)
        print()

    print("=" * 60)
    print("✅ Done.")
    print("=" * 60)

if __name__ == "__main__":
    main()