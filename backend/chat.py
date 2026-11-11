"""Rule-based research chatbot (no API keys).
1) Questions about the analyzed paper -> intent detection -> answer built from its analysis.
2) Questions about research papers / NLP -> built-in glossary.
3) Anything else -> Wikipedia summary + related papers from open scholarly APIs."""
import re, requests
from . import nlp, search
from .knowledge import G

PAPER_CUE = re.compile(r"\b(this|the|that|my|uploaded|analy[sz]ed)\s+(paper|work|study|article|research|authors?)\b|\bthe authors?\b|\bin the paper\b|\bused\b|\bthey\b|\bproposed\b", re.I)
GENERAL_CUE = re.compile(r"^(what is|what's|what are)\s+(a|an)\s|\bdefine\b|\bmeaning of\b|\bdifferences?\b|\bcompare\b|\bversus\b|\bvs\.?\b|\brecent\b|\btrends?\b|\blatest\b|\bstate of the art\b|\bevaluate\b|\bhow to\b|\bwhy (do|is)\b", re.I)
INTENTS = [
 ("gap", r"research gap|\bgaps?\b"),
 ("future", r"future|improve|overcome|suggest|recommend|next step|extend|enhance"),
 ("limits", r"limitation|drawback|weakness|shortcoming|disadvantage|lacks?\b"),
 ("data", r"data ?sets?|corpus|corpora|training data|benchmark"),
 ("metrics", r"metric|measure|evaluat|score"),
 ("results", r"result|accuracy|performance|outcome|finding|achiev|how (well|good)"),
 ("concl", r"conclu"),
 ("methods", r"method|model|algorithm|technique|approach|architecture|framework|how (does|did|do)\b|proposed"),
 ("summary", r"summar|overview|explain|tl;?dr|gist|in short|main idea|about (this|the) paper|paper about|what does (this|the) paper"),
 ("problem", r"problem|objective|\baims?\b|goal|motivation|purpose|address"),
 ("authors", r"authors?|who (wrote|published)|written by"),
 ("title", r"\btitle\b|name of (the|this) paper"),
 ("dates", r"dates?\b|when (was|is)|published|received|accepted"),
 ("keywords", r"keywords?|key ?terms|keyphrases?"),
 ("entities", r"organi[sz]ation|institution|universit|countr|location|people|entities|tools|software|libraries"),
]
UA = {"User-Agent": "ResearchAssistant/1.0 (student project)"}

def _b(title, items):
    items = [i for i in items if i]
    return f"**{title}**\n" + ("\n".join("- " + i for i in items) if items else "- Not clearly stated in the text.")

def _clip(s, n=260):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

def _intents(q):
    return [n for n, rx in INTENTS if re.search(rx, q, re.I)][:3]

def _paper(q, S, names):
    A = S["analysis"]; k, ex, gp = A["key_info"], A["extraction"], A["gap_analysis"]
    B, M, N = k["Basics"], k["Models, data & metrics"], k["Named entities"]
    blocks = []
    for n in names:
        blocks.append({
         "gap": lambda: _b("Research gap", gp["Limitations we noticed"][:4] + gp["Limitations stated in the paper"][:2]),
         "future": lambda: _b("Suggestions / future work", gp["Suggestions to overcome"][:6]),
         "limits": lambda: _b("Limitations", gp["Limitations stated in the paper"][:3] + gp["Limitations we noticed"][:4]),
         "data": lambda: _b("Datasets", M["Datasets"][:8]),
         "metrics": lambda: _b("Evaluation metrics", M["Metrics"][:10]),
         "results": lambda: _b("Main results", ex["Results"] + ["Metrics reported: " + ", ".join(M["Metrics"][:6])] * bool(M["Metrics"])),
         "concl": lambda: _b("Conclusion", ex["Conclusion"]),
         "methods": lambda: _b("Methods and models", ex["Methods used"] + ["Models detected: " + ", ".join(M["Models / methods"][:8])] * bool(M["Models / methods"])),
         "summary": lambda: _b("Summary", nlp.sentences(A["summary"])[:5] or [A["summary"]]),
         "problem": lambda: _b("Problem addressed", ex["Problem"]),
         "authors": lambda: _b("Authors", B["Authors"]),
         "title": lambda: _b("Title", [B["Title"]]),
         "dates": lambda: _b("Important dates", B["Important dates"]),
         "keywords": lambda: _b("Keywords", B["Keywords"]),
         "entities": lambda: _b("Named entities", ["Organizations: " + ", ".join(N["Organizations"][:6]) if N["Organizations"] else "",
                                                   "Locations: " + ", ".join(N["Locations"][:6]) if N["Locations"] else "",
                                                   "Products / tools: " + ", ".join(N["Products"][:8]) if N["Products"] else ""]),
        }[n]())
    return {"answer": f"**From your paper:** “{S['title'] or 'analyzed paper'}”\n\n" + "\n\n".join(blocks), "sources": []}

def _glossary(q):
    hits = []
    for rx, title, d, pts in G:
        m = re.search(rx, q, re.I)
        if m: hits.append((m.start(), title, d, pts))
    hits.sort(key=lambda h: h[0])
    return hits[:3]

def _wiki(term):
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=8, params={
            "action": "query", "generator": "search", "gsrsearch": term, "gsrlimit": 1, "prop": "extracts|info",
            "inprop": "url", "exintro": 1, "explaintext": 1, "exsentences": 3, "format": "json"})
        pg = next(iter(r.json()["query"]["pages"].values()))
        return pg["title"], _clip(pg.get("extract"), 520), pg["fullurl"]
    except Exception:
        return None

def _topic(q):
    t = re.sub(r"[?.!]+$", "", q.strip())
    t = re.sub(r"^(please\s+)?(tell me about|explain|define|what(?:'s| is| are| was)|who is|which|how (?:do|does|to)|why (?:is|do|does))\s+(the\s+)?", "", t, flags=re.I)
    t = re.sub(r"\b(research )?papers?( on| about| for)?\b|\b(recent|latest|trends?( in)?|state of the art( in)?|survey( of| on)?|review( of| on)?)\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip() or q

def _papers(topic, n=3, y1=2020):
    try:
        res = search.search(topic, y1, 2026, 2, ["openalex", "arxiv"])
    except Exception:
        return []
    return [p for p in res if p.get("title")][:n]

def _plist(ps):
    return [f"{_clip(p['title'], 110)} ({p.get('year') or '—'}, {p['source']})" + (f" — {_clip(nlp.summarize(p['abstract'], 1), 170)}" if len(p.get("abstract") or "") > 80 else "") for p in ps]

def _src(ps): return [{"title": p["title"], "url": p["url"]} for p in ps if p.get("url")]

def _general(q):
    hits = _glossary(q)
    wants_papers = re.search(r"recent|latest|trends?|state of the art|advances|developments|\bfind\b|\bshow\b|\blist\b|papers? (on|about|for)", q, re.I)
    blocks, sources = [], []
    if hits and not wants_papers:
        for _, t, d, pts in hits:
            blocks.append(f"**{t}**\n- {d}\n" + "\n".join("- " + p for p in pts))
        if len(hits) == 2 and re.search(r"difference|compare|versus|vs\.?", q, re.I):
            blocks.append("**Tip**\n- Compare the two on purpose, input/output, data needed and typical use.")
        return {"answer": "\n\n".join(blocks), "sources": []}
    topic = _topic(q)
    m = re.search(r"(?:between|compare)\s+(.+?)\s+(?:and|vs\.?|versus|with|&)\s+(.+)", q.rstrip("?. "), re.I) or re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", q.rstrip("?. "), re.I)
    if m:
        for term in m.groups():
            w = _wiki(term.strip())
            if w: blocks.append(f"**{w[0]}**\n- {w[1]}"); sources.append({"title": w[0] + " (Wikipedia)", "url": w[2]})
    elif not wants_papers:
        w = _wiki(topic)
        if w: blocks.append(f"**{w[0]}**\n- {w[1]}"); sources.append({"title": w[0] + " (Wikipedia)", "url": w[2]})
    ps = _papers(topic, 3, 2022 if wants_papers else 2019)
    if ps:
        blocks.append("**Recent research papers**\n" + "\n".join("- " + x for x in _plist(ps))); sources += _src(ps)
    if not blocks:
        return {"answer": "I could not find a clear answer. Try rewording with the key term (for example “What is k-fold cross-validation?”), or check your internet connection.", "sources": []}
    return {"answer": "\n\n".join(blocks), "sources": sources}

def answer(q, S):
    q = q.strip()
    if re.fullmatch(r"(hi|hello|hey|thanks|thank you|ok|okay)\W*", q, re.I):
        return {"answer": "Hello! Ask me about your analyzed paper (problem, methods, datasets, results, limitations, future work) or about research papers and NLP (for example “What is a research gap?”).", "sources": []}
    names, has = _intents(q), bool(S.get("analysis"))
    cue, gen = PAPER_CUE.search(q), GENERAL_CUE.search(q)
    if names and (cue or (not gen and not _glossary(q))):
        if has: return _paper(q, S, names)
        out = _general(q)
        out["answer"] = "**No paper analyzed yet.** Upload a PDF in *Analyze Paper* to get answers about your own paper. Meanwhile, general information:\n\n" + out["answer"]
        return out
    if has and not _glossary(q) and not gen and not re.search(r"\bpapers?\b", q, re.I):
        ctx = nlp.retrieve(q, S["text"])
        if ctx: return {"answer": _b("Closest passages in your paper", [_clip(c, 300) for c in ctx]), "sources": []}
    return _general(q)
