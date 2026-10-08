"""Hallucination detection layer for FinRAG."""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MODEL_NAME = "gemini-3.8-flash"

VERIFICATION_PROMPT = """You are a fact-verification assistant. You will be given:
1. A CLAIM (an answer produced by another AI)
2. The SOURCES it was supposed to be based on

Your job: verify EVERY factual claim in the CLAIM against the SOURCES.

Rules:
- A claim is "supported" only if the exact fact (or a direct paraphrase) appears in the sources.
- Numbers must match exactly. "$391B" is different from "$391.035B".
- If a claim is not found in the sources, mark it as "unsupported".
- Do not use any outside knowledge.

Output format (strict — follow exactly):
VERDICT: [SUPPORTED | PARTIAL | UNSUPPORTED]
SUPPORTED_CLAIMS:
  - <claim 1>
  - <claim 2>
UNSUPPORTED_CLAIMS:
  - <claim X> — reason: <why it's not supported>
CONFIDENCE: [HIGH | MEDIUM | LOW]

SOURCES:
{context}

CLAIM:
{answer}

VERIFICATION:"""

def build_verifier():
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )
    prompt = ChatPromptTemplate.from_template(VERIFICATION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    return chain

def format_sources(sources):
    formatted = []
    for i, doc in enumerate(sources, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        formatted.append(f"[Source {i} — {source}, page {page}]\n{content}")
    return "\n\n".join(formatted)

def verify_answer(answer, sources, verifier=None):
    if verifier is None:
        verifier = build_verifier()
    context = format_sources(sources)
    verdict = verifier.invoke({"context": context, "answer": answer})
    return verdict

def parse_verdict(verdict_text):
    lines = verdict_text.strip().split("\n")
    result = {"verdict": "UNKNOWN", "supported": [], "unsupported": [], "confidence": "UNKNOWN"}

    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("VERDICT:"):
            result["verdict"] = line.replace("VERDICT:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip()
        elif line.startswith("SUPPORTED_CLAIMS:"):
            current_section = "supported"
        elif line.startswith("UNSUPPORTED_CLAIMS:"):
            current_section = "unsupported"
        elif line.startswith("- ") and current_section:
            result[current_section].append(line[2:])

    return result

if __name__ == "__main__":
    print("Verification module loaded.")