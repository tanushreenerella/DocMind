"""
Generates 5 reproducible sample PDFs in the sample_docs/ directory.
Run from the project root: python scripts/create_sample_docs.py
Requires: pip install reportlab
"""
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT_DIR = Path(__file__).parent.parent / "sample_docs"
OUT_DIR.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=4)
BODY = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, textColor=colors.grey)


def build(filename: str, story: list, pagesize=A4) -> None:
    path = OUT_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=pagesize, leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    doc.build(story)
    print(f"  Created {path}")


# ── 1. Invoice ────────────────────────────────────────────────────────────────
def invoice():
    story = [
        Paragraph("INVOICE", H1),
        Paragraph("ACME Corporation · 123 Business Ave, San Francisco, CA 94105", SMALL),
        Paragraph("Invoice #: INV-2024-0042 &nbsp;&nbsp; Date: 2024-11-15 &nbsp;&nbsp; Due: 2024-12-15", BODY),
        HRFlowable(width="100%", thickness=1, color=colors.lightgrey),
        Spacer(1, 0.2 * inch),
        Paragraph("Bill To:", H2),
        Paragraph("Beta Startup Inc.<br/>456 Tech Blvd, Austin, TX 78701", BODY),
        Spacer(1, 0.2 * inch),
    ]

    data = [
        ["Description", "Qty", "Unit Price", "Total"],
        ["Cloud Infrastructure Setup", "1", "$4,500.00", "$4,500.00"],
        ["LLM Integration Consulting (hrs)", "20", "$250.00", "$5,000.00"],
        ["Vector Database License (annual)", "1", "$1,200.00", "$1,200.00"],
        ["Technical Documentation", "3", "$400.00", "$1,200.00"],
        ["Support & Maintenance (3 months)", "1", "$900.00", "$900.00"],
        ["", "", "Subtotal", "$12,800.00"],
        ["", "", "Tax (8.5%)", "$1,088.00"],
        ["", "", "TOTAL DUE", "$13,888.00"],
    ]
    t = Table(data, colWidths=[3.2 * inch, 0.8 * inch, 1.2 * inch, 1.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -2), 0.5, colors.lightgrey),
        ("FONTNAME", (2, -3), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (2, -1), (-1, -1), colors.HexColor("#EEF2FF")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -4), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    story.extend([t, Spacer(1, 0.3 * inch),
                  Paragraph("Payment Terms: Net 30 days. Wire transfer details on file.", SMALL)])
    build("invoice_acme_2024.pdf", story)


# ── 2. Research paper ─────────────────────────────────────────────────────────
def research_paper():
    story = [
        Paragraph("Advances in Retrieval-Augmented Generation for Enterprise Document Intelligence", H1),
        Paragraph("Abstract", H2),
        Paragraph(
            "Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for grounding "
            "large language model (LLM) outputs in verified external knowledge. This paper surveys "
            "recent architectural improvements including hybrid dense-sparse retrieval, reranking "
            "strategies, and agentic multi-step reasoning chains. We evaluate these on a novel "
            "enterprise benchmark comprising 10,000 heterogeneous documents across six domain "
            "verticals and report RAGAS scores alongside human preference ratings. Our best system "
            "achieves a faithfulness score of 0.91 and an answer relevancy of 0.87.",
            BODY,
        ),
        Spacer(1, 0.15 * inch),
        Paragraph("1. Introduction", H2),
        Paragraph(
            "Enterprise knowledge management suffers from information fragmentation: critical facts "
            "are buried in PDFs, spreadsheets, internal wikis, and email threads. LLMs alone "
            "hallucinate when queried beyond their training distribution. RAG bridges this gap by "
            "decomposing a query into a retrieval step and a generation step, conditioning the model "
            "on retrieved passages at inference time (Lewis et al., 2020).",
            BODY,
        ),
        Paragraph("2. Related Work", H2),
        Paragraph(
            "Early RAG implementations relied on BM25 sparse retrieval. DPR (Karpukhin et al., 2020) "
            "introduced dense passage retrieval with bi-encoder architectures, achieving significant "
            "gains on open-domain QA. Subsequent work (Izacard & Grave, 2021) demonstrated that "
            "fusion-in-decoder approaches outperform single-passage generation by attending over "
            "multiple retrieved documents simultaneously.",
            BODY,
        ),
        Paragraph("3. Architecture", H2),
        Paragraph(
            "Our system comprises four stages: (1) document ingestion with OCR fallback for scanned "
            "pages, (2) semantic chunking with 500-character windows and 50-character overlap, "
            "(3) hybrid retrieval combining cosine similarity on MiniLM-L6 embeddings with BM25 "
            "keyword matching, and (4) LLaMA-3 70B generation with citation-aware prompting. "
            "ChromaDB provides persistent vector storage with HNSW indexing for sub-100ms retrieval "
            "at 1M+ chunk scale.",
            BODY,
        ),
        Paragraph("4. Results", H2),
    ]

    data = [
        ["Model", "Faithfulness", "Answer Relevancy", "Context Recall"],
        ["BM25 + GPT-3.5", "0.71", "0.74", "0.68"],
        ["DPR + GPT-4", "0.83", "0.81", "0.79"],
        ["Hybrid + LLaMA-3 70B (ours)", "0.91", "0.87", "0.88"],
    ]
    t = Table(data, colWidths=[2.5 * inch, 1.2 * inch, 1.4 * inch, 1.3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6D28D9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F3FF")]),
    ]))
    story.extend([t, Spacer(1, 0.15 * inch),
                  Paragraph("Table 1: RAGAS evaluation on the enterprise benchmark.", SMALL),
                  Spacer(1, 0.2 * inch),
                  Paragraph("5. Conclusion", H2),
                  Paragraph(
                      "Hybrid retrieval combined with citation-aware generation substantially improves "
                      "factual grounding in enterprise RAG systems. Future work will explore "
                      "multi-hop reasoning and dynamic knowledge graph integration.", BODY)])
    build("research_paper_ai.pdf", story)


# ── 3. Medical report ─────────────────────────────────────────────────────────
def medical_report():
    story = [
        Paragraph("CONFIDENTIAL MEDICAL REPORT", H1),
        Paragraph("City General Hospital · Department of Internal Medicine", SMALL),
        HRFlowable(width="100%", thickness=1, color=colors.lightgrey),
        Spacer(1, 0.15 * inch),
        Paragraph("Patient Information", H2),
        Paragraph("Name: Jane Doe &nbsp;&nbsp; DOB: 1985-03-22 &nbsp;&nbsp; MRN: 10294857", BODY),
        Paragraph("Date of Visit: 2024-10-08 &nbsp;&nbsp; Attending: Dr. Sarah Patel, MD", BODY),
        Paragraph("Chief Complaint", H2),
        Paragraph("Patient presents with persistent fatigue, mild shortness of breath on exertion, "
                  "and occasional palpitations over the past 6 weeks.", BODY),
        Paragraph("Examination Findings", H2),
        Paragraph("BP: 128/82 mmHg · HR: 88 bpm · RR: 16/min · SpO2: 97% · Temp: 36.8°C", BODY),
        Paragraph("Cardiovascular: Regular rhythm, no murmurs. Respiratory: Clear to auscultation "
                  "bilaterally. Abdomen: Soft, non-tender.", BODY),
        Paragraph("Laboratory Results", H2),
    ]

    data = [
        ["Test", "Result", "Reference Range", "Flag"],
        ["Hemoglobin", "10.2 g/dL", "12.0–16.0 g/dL", "LOW ↓"],
        ["Ferritin", "8 ng/mL", "12–150 ng/mL", "LOW ↓"],
        ["TSH", "2.1 mIU/L", "0.4–4.0 mIU/L", "Normal"],
        ["B12", "310 pg/mL", "200–900 pg/mL", "Normal"],
        ["CRP", "0.8 mg/L", "< 1.0 mg/L", "Normal"],
    ]
    t = Table(data, colWidths=[2 * inch, 1.2 * inch, 1.8 * inch, 1.4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TEXTCOLOR", (3, 1), (3, 2), colors.red),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDFA")]),
    ]))
    story.extend([t, Spacer(1, 0.2 * inch),
                  Paragraph("Diagnosis", H2),
                  Paragraph("Iron-deficiency anaemia (ICD-10: D50.9). Likely dietary in origin.", BODY),
                  Paragraph("Plan", H2),
                  Paragraph("1. Oral ferrous sulphate 200 mg TDS for 3 months.\n"
                             "2. Dietary counselling for iron-rich foods.\n"
                             "3. Repeat CBC and ferritin in 8 weeks.\n"
                             "4. Consider gastroenterology referral if no improvement.", BODY)])
    build("medical_report.pdf", story)


# ── 4. Financial summary ──────────────────────────────────────────────────────
def financial_summary():
    story = [
        Paragraph("Q3 2024 Financial Summary Report", H1),
        Paragraph("TechVentures Inc. · Board of Directors · Confidential", SMALL),
        HRFlowable(width="100%", thickness=1, color=colors.lightgrey),
        Spacer(1, 0.15 * inch),
        Paragraph("Executive Summary", H2),
        Paragraph(
            "Q3 2024 delivered record revenue of $47.2M, a 34% YoY increase, driven by strong "
            "enterprise SaaS expansion and successful launch of our AI product tier. Operating "
            "expenses rose 18% to $31.4M, resulting in operating income of $15.8M (33.5% margin). "
            "ARR crossed $180M, with net revenue retention at 118%.",
            BODY,
        ),
        Paragraph("Key Financial Metrics", H2),
    ]

    data = [
        ["Metric", "Q3 2024", "Q3 2023", "YoY Change"],
        ["Revenue", "$47.2M", "$35.2M", "+34%"],
        ["Gross Profit", "$33.0M", "$23.6M", "+40%"],
        ["Gross Margin", "70.0%", "67.0%", "+3pp"],
        ["Operating Income", "$15.8M", "$9.2M", "+72%"],
        ["Operating Margin", "33.5%", "26.1%", "+7.4pp"],
        ["ARR", "$180.3M", "$128.7M", "+40%"],
        ["Net Revenue Retention", "118%", "112%", "+6pp"],
        ["Headcount", "642", "498", "+29%"],
    ]
    t = Table(data, colWidths=[2.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TEXTCOLOR", (3, 1), (3, -1), colors.HexColor("#15803D")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EFF6FF")]),
    ]))
    story.extend([t, Spacer(1, 0.2 * inch),
                  Paragraph("Segment Breakdown", H2),
                  Paragraph("Enterprise (≥500 seats): $28.3M (+45% YoY) — 60% of revenue. "
                             "SMB (50–499 seats): $12.8M (+22% YoY). "
                             "Self-serve (<50 seats): $6.1M (+18% YoY).", BODY),
                  Paragraph("Outlook", H2),
                  Paragraph("FY2024 guidance raised to $183–186M revenue (prev. $175–178M). "
                             "Q4 pipeline is $62M with 74% close-rate confidence.", BODY)])
    build("financial_summary.pdf", story)


# ── 5. Technical manual ───────────────────────────────────────────────────────
def technical_manual():
    story = [
        Paragraph("DocMind API — Technical Integration Manual", H1),
        Paragraph("Version 1.0 · Engineering Team · Internal Use Only", SMALL),
        HRFlowable(width="100%", thickness=1, color=colors.lightgrey),
        Spacer(1, 0.15 * inch),
        Paragraph("1. Overview", H2),
        Paragraph(
            "DocMind exposes a RESTful API for document ingestion, semantic search, and "
            "AI-powered question answering. The API is built on FastAPI and follows OpenAPI 3.1 "
            "specifications. All endpoints require a valid API key passed in the X-API-Key header.",
            BODY,
        ),
        Paragraph("2. Authentication", H2),
        Paragraph("Requests without a valid X-API-Key return HTTP 401. Keys are issued per "
                  "workspace and rotated every 90 days. Store keys in environment variables — "
                  "never hard-code them in source code.", BODY),
        Paragraph("3. Endpoints", H2),
    ]

    data = [
        ["Method", "Endpoint", "Description"],
        ["POST", "/api/upload", "Upload one or more documents (multipart/form-data)"],
        ["GET", "/api/status/{doc_id}", "Poll processing status and classification result"],
        ["POST", "/api/chat", "Submit a question with conversation history"],
        ["POST", "/api/transcribe", "Transcribe base64-encoded audio via Whisper"],
        ["GET", "/api/documents", "List all indexed documents with metadata"],
        ["GET", "/api/page-image/{filename}", "Serve a rendered page image (JPEG)"],
        ["GET", "/health", "Liveness check — returns {status: ok}"],
    ]
    t = Table(data, colWidths=[0.8 * inch, 2.4 * inch, 3.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTNAME", (0, 1), (1, -1), "Courier"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    story.extend([t, Spacer(1, 0.2 * inch),
                  Paragraph("4. Rate Limits", H2),
                  Paragraph("Default: 30 requests/minute per IP. Exceeded requests return "
                             "HTTP 429 with a Retry-After header. Contact support to raise limits.", BODY),
                  Paragraph("5. Error Codes", H2),
                  Paragraph("400 Bad Request — invalid file type or input. "
                             "404 Not Found — doc_id or image not found. "
                             "429 Too Many Requests — rate limit exceeded. "
                             "500 Internal Server Error — processing failure (retry safe).", BODY)])
    build("technical_manual.pdf", story)


if __name__ == "__main__":
    print("Generating sample documents…")
    invoice()
    research_paper()
    medical_report()
    financial_summary()
    technical_manual()
    print("Done! All files written to sample_docs/")
