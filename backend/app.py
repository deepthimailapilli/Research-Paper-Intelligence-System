import math, re
from pathlib import Path
from typing import List
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import nlp, search, chat as chatbot, compare as cmp, citations, report as rpt

MAX_MB = 200
app = FastAPI(title="Research Assistant")
STATE = {"text": "", "secs": {}, "keywords": [], "title": "", "analysis": None}
FRONT = Path(__file__).resolve().parent.parent / "frontend"

def build(text, pages, note="", secs=None, full=True):
    """One analysis used everywhere. full=False returns only the research-gap block (Find Papers)."""
    secs = secs or nlp.split_sections(text)
    ki, out = nlp.key_info(text), {"pages": pages, "note": note, "gap_analysis": nlp.gap_analysis(text, secs)}
    if full:
        out.update(summary=nlp.summarize(text, 5),
                   section_summaries={k: nlp.summarize(v, 3) for k, v in secs.items()},
                   key_info=ki, extraction=nlp.extraction(text, secs))
        # Chat and Recommendations follow the paper analyzed in the Analyze Paper module
        STATE.update(text=text, secs=secs, keywords=ki["Basics"]["Keywords"], title=ki["Basics"]["Title"], analysis=out)
    return out

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024: raise HTTPException(413, f"File exceeds {MAX_MB} MB")
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(400, "Upload a PDF file")
    try: text, pages = nlp.extract_text(data)
    except Exception: raise HTTPException(400, "Could not read this PDF")
    if len(text.strip()) < 300: raise HTTPException(400, "No readable text (scanned PDF?)")
    return build(text, pages)

class P(BaseModel):
    title: str = ""
    abstract: str = ""
    pdf_url: str = ""

@app.post("/api/analyze-paper")
def analyze_paper(p: P):
    """Analyze a paper from Find Papers: full PDF if free, otherwise the abstract."""
    if p.pdf_url.startswith(("http://", "https://")):
        try:
            r = requests.get(p.pdf_url, timeout=40, stream=True, headers={"User-Agent": "Mozilla/5.0"})
            buf = b""
            for chunk in r.iter_content(1 << 20):
                buf += chunk
                if len(buf) > MAX_MB * 1024 * 1024: raise ValueError("too large")
            if buf[:5] == b"%PDF-":
                text, pages = nlp.extract_text(buf)
                if len(text.strip()) >= 300: return build(text, pages, full=False)
        except Exception:
            pass
    if len(p.abstract.strip()) < 50:
        raise HTTPException(400, "Full PDF is not free and this paper has no abstract to analyze.")
    return build(p.title + "\n" + p.abstract, 0,
                 "Full PDF was not freely available, so this analysis uses the abstract only.",
                 {"Abstract": p.abstract}, full=False)

@app.get("/api/find")
def find(topic: str, year_from: int = 2018, year_to: int = 2026, count: int = 2,
         sources: str = "ieee,springer,elsevier,acm,openalex,arxiv"):
    return {"results": search.search(topic, year_from, year_to, max(1, min(count, 10)), sources.split(","))}

# ---------- compare papers / literature review ----------
@app.post("/api/compare")
async def compare_papers(files: List[UploadFile] = File(...)):
    """Analyze 2-3 PDFs side by side. Does not change the paper used by Chat / Recommendations."""
    if not 2 <= len(files) <= 3: raise HTTPException(400, "Upload 2 or 3 PDF files to compare")
    papers = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_MB * 1024 * 1024: raise HTTPException(413, f"{f.filename} exceeds {MAX_MB} MB")
        if not (f.filename or "").lower().endswith(".pdf"): raise HTTPException(400, f"{f.filename}: upload PDF files only")
        try: text, pages = nlp.extract_text(data)
        except Exception: raise HTTPException(400, f"Could not read {f.filename}")
        if len(text.strip()) < 300: raise HTTPException(400, f"{f.filename}: no readable text (scanned PDF?)")
        papers.append(cmp.profile(text, pages, f.filename))
    return {"papers": papers, "rows": cmp.rows(papers)}

class LR(BaseModel):
    papers: List[dict]

@app.post("/api/litreview")
def lit_review(b: LR):
    """Draft a related-work paragraph from compared papers or analyses saved in the Library."""
    if len(b.papers) < 2: raise HTTPException(400, "Select at least 2 analyzed papers")
    if len(b.papers) > 12: raise HTTPException(400, "Select at most 12 papers")
    try: return cmp.review(b.papers)
    except (KeyError, TypeError, ValueError): raise HTTPException(400, "Could not read the selected analyses")

# ---------- report download ----------
class Rep(BaseModel):
    kind: str = "analysis"      # analysis | compare | review
    format: str = "pdf"         # pdf | docx
    title: str = ""
    notes: str = ""
    data: dict = {}

@app.post("/api/report")
def make_report(r: Rep):
    fmt = r.format.lower()
    if fmt not in ("pdf", "docx"): raise HTTPException(400, "Format must be pdf or docx")
    try:
        blocks = rpt.blocks(r.kind, r.data, r.title, r.notes)
        body = rpt.to_pdf(blocks) if fmt == "pdf" else rpt.to_docx(blocks)
    except ImportError:
        raise HTTPException(500, "Report libraries are missing. Run: pip install -r requirements.txt")
    except ValueError as e:
        raise HTTPException(400, str(e))
    name = re.sub(r"[^a-z0-9]+", "-", (r.title or r.kind).lower()).strip("-")[:60] or "report"
    mime = "application/pdf" if fmt == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return Response(body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"'})

class Q(BaseModel):
    question: str

@app.post("/api/chat")
def chat(q: Q):
    return chatbot.answer(q.question, STATE)

@app.get("/api/recommend")
def recommend(count: int = 10, year_from: int = 2015, year_to: int = 2026):
    if not STATE["keywords"]: raise HTTPException(400, "Analyze a paper first")
    count = max(1, min(count, 30))
    res = search.search(" ".join(STATE["keywords"][:4]), year_from, year_to, math.ceil(count / 6) + 1)
    res = [r for r in res if r["title"] and r["title"].lower() not in STATE["title"].lower()][:count]
    return {"based_on": STATE["keywords"][:4], "based_on_title": STATE["title"], "results": res}

M = {"A": "Analyze Paper", "F": "Find Papers", "C": "Chat", "R": "Recommendations", "P": "Compare Papers", "L": "Library"}
ABOUT = [
 ("Unit 1", "Introduction to NLP and the NLP pipeline", [
  ("NLP tasks and applications", "Summarization, information extraction, question answering, retrieval and recommendation of research papers", "AFCR", "all modules"),
  ("Building blocks of language", "Text is split into sections, sentences and words / phrases", "A", "nlp.split_sections, nlp.sentences"),
  ("Heuristics-based NLP", "Regex section headings and keyword rules for problem, methods, results and limitation checks", "A", "nlp.CHECKS, nlp.extraction"),
  ("Machine learning for NLP", "TF-IDF features with cosine similarity (scikit-learn)", "ACR", "nlp.summarize, nlp.retrieve"),
  ("NLP pipeline: data acquisition", "PDF upload and open scholarly APIs (Crossref, OpenAlex, arXiv)", "AF", "app.analyze, search.search"),
  ("Text extraction and cleanup", "pypdf text extraction; whitespace and tag cleaning of abstracts", "AF", "nlp.extract_text, search._c"),
  ("Pre-processing", "Sentence segmentation, tokenization, lower-casing and stop-word removal", "AC", "nlp.sentences, TfidfVectorizer"),
  ("Feature engineering", "TF-IDF weights on unigrams and bigrams", "AC", "nlp.keywords"),
  ("Building the model", "TextRank-style summarizer, rule-based extractor, retrieval ranker", "AC", "nlp.summarize, nlp.extraction")]),
 ("Unit 2", "Text representation", [
  ("Vector space model", "Every sentence, question and paper becomes a TF-IDF vector", "ACR", "TfidfVectorizer"),
  ("Basic vectorization approaches", "Weighted bag of words with 1-2 word n-grams", "AC", "nlp.keywords, nlp.retrieve"),
  ("Similarity in vector space", "Cosine similarity ranks sentences for summaries and answers", "AC", "nlp.summarize")]),
 ("Unit 3", "Text classification", [
  ("Pipeline for text classification", "Each sentence is assigned to a class (problem, method, result, conclusion, limitation) using pattern rules", "A", "nlp.extraction, nlp.gap_analysis"),
  ("Learning with no or less data", "No labelled data or training is needed, so the tool adapts to any research domain", "AF", "rule-based + unsupervised TF-IDF")]),
 ("Unit 4", "Information extraction", [
  ("IE applications and tasks", "A paper is turned into a structured card: title, authors, dates, models, datasets, metrics", "A", "nlp.key_info"),
  ("General IE pipeline", "Sections -> sentences -> entity patterns -> structured output", "A", "nlp.key_info, nlp.extraction"),
  ("Keyphrase extraction", "Top TF-IDF words and bigrams become the keywords", "AR", "nlp.keywords"),
  ("Named entity recognition", "People, organizations, locations, products, dates plus models, datasets, databases and metrics", "A", "nlp.key_info"),
  ("Building an NER system", "Gazetteer lists + regex patterns, no external NER library", "A", "nlp.MODELS, ORGS, LOCS, PRODUCTS"),
  ("Relationship extraction", "Cited authors are found from 'Author et al., year' patterns; dates are linked to Received / Accepted labels", "A", "nlp._dates, nlp.key_info"),
  ("Comparing extracted records", "Methods, datasets, metrics, results and limitations of 2-3 papers are aligned in one table and shared items are highlighted", "P", "compare.profile, compare.rows")]),
 ("Unit 5", "Chatbots", [
  ("Chatbot applications and taxonomy", "A closed-domain, retrieval-based research assistant bot", "C", "chat.answer"),
  ("Pipeline for building dialog systems", "Question -> intent detection -> retrieval -> formatted reply", "C", "chat.answer"),
  ("Components of a dialog system", "Intent recognizer, knowledge sources (analyzed paper, glossary, web) and reply templates", "C", "chat._intents, knowledge.G")]),
 ("Unit 6", "Brief overview of NLP applications", [
  ("Information retrieval and search components", "Query -> multi-source retrieval -> de-duplication -> ranking", "FR", "search.search"),
  ("Text summarization", "Extractive summary of the paper and of each section", "A", "nlp.summarize"),
  ("Recommender systems for textual data", "Content-based: similar papers found from the paper's keywords", "R", "app.recommend"),
  ("Question answering", "Answers from the analyzed paper, built-in glossary and open sources", "C", "chat.answer"),
  ("Template-based text generation", "A related-work paragraph is drafted from the facts extracted from each paper", "P", "compare.review")]),
]

@app.get("/api/about")
def about():
    units = [{"unit": u, "title": t, "topics": [{"topic": a, "how": b, "where": [M[c] for c in w], "code": code}
                                               for a, b, w, code in ts]} for u, t, ts in ABOUT]
    return {"units": units, "topic_count": sum(len(u["topics"]) for u in units)}

@app.get("/")
def index(): return FileResponse(FRONT / "index.html")
app.mount("/static", StaticFiles(directory=FRONT), name="static")
