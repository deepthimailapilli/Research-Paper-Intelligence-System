"""Compare 2-3 papers side by side and draft a related-work paragraph from them.
Everything is rule-based (templates over the facts nlp.py already extracts), so no API key or model is needed."""
import re
from collections import Counter
from . import nlp, citations

COUNT_RX = re.compile(r"\s*×\d+\s*$")
VERBS = ["employed", "used", "adopted"]
GENERIC_KW = {"model", "models", "paper", "study", "approach", "method", "methods", "results", "data", "proposed", "based", "using", "work", "dataset", "datasets", "learning", "performance"}
RESULT_RX = re.compile(r"\b(accuracy|f1(?:[- ]?score)?|precision|recall|auc|bleu|rouge(?:-[l\d])?|rmse|mae)\b[^.\d%]{0,40}?(\d{1,3}(?:\.\d+)?)\s?(%?)", re.I)
AN = {"F1", "RMSE", "MAE"}


def _clip(s, n=260):
    s = re.sub(r"\s+", " ", s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _names(items):
    """['BERT ×3', 'LSTM ×1'] -> ['BERT', 'LSTM'] (counts removed so papers can be compared)."""
    return list(dict.fromkeys(COUNT_RX.sub("", x).strip() for x in items or [] if x and x.strip()))


def _year(dates, head):
    ys = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-2]\d)\b", " ".join(dates) or head)]
    ys = [y for y in ys if y <= 2027]
    return str(max(ys)) if ys else ""


def _build(name, pages, ki, ex, gap, head="", title=""):
    b, m = ki["Basics"], ki["Models, data & metrics"]
    return {"name": name, "pages": pages, "title": title or b["Title"], "authors": b["Authors"],
            "year": _year(b["Important dates"], head), "keywords": b["Keywords"],
            "problem": ex.get("Problem", []), "methods": _names(m["Models / methods"]),
            "method_notes": ex.get("Methods used", []), "datasets": _names(m["Datasets"]),
            "metrics": _names(m["Metrics"]), "results": ex.get("Results", []),
            "conclusion": ex.get("Conclusion", []),
            "limitations_stated": gap.get("Limitations stated in the paper", []),
            "limitations_noticed": gap.get("Limitations we noticed", [])}


def profile(text, pages, name=""):
    """Analyze one paper's text WITHOUT touching the app-wide chat state."""
    secs = nlp.split_sections(text)
    return _build(name, pages, nlp.key_info(text), nlp.extraction(text, secs), nlp.gap_analysis(text, secs), head=text[:3000])


def from_analysis(title, d):
    """Build a profile from an analysis saved in the Library (full analysis or Find-Papers gap analysis)."""
    gap = d.get("gap_analysis") or {}
    ki = d.get("key_info") or {"Basics": {"Title": title, "Authors": [], "Important dates": [], "Keywords": []},
                               "Models, data & metrics": {"Models / methods": [], "Datasets": [], "Metrics": []}}
    ex = d.get("extraction") or {"Problem": gap.get("Problem statement", []), "Methods used": gap.get("Methods used", []),
                                 "Results": [], "Conclusion": []}
    return _build(title or ki["Basics"]["Title"], d.get("pages", 0), ki, ex, gap, title=title)


def _norm(item):
    return from_analysis(item.get("title", ""), item["analysis"]) if "analysis" in item else item


# ---------- comparison table ----------
def _short(items):
    return [x for x in items if len(x) <= 40], [x for x in items if len(x) > 40]


def rows(papers):
    """Rows of the comparison table. Each cell has chips (short items) and lists (sentences)."""
    def chip_row(label, key):
        cells, allc = [], []
        for p in papers:
            short, long_ = _short(p.get(key, []))
            cells.append({"chips": short[:10], "lists": [{"label": None, "items": [_clip(x, 240) for x in long_[:2]]}]})
            allc.append({x.lower() for x in short})
        shared = sorted(x for x, n in Counter(c for s in allc for c in s).items() if n >= 2)
        return {"label": label, "cells": cells, "shared": shared}

    def list_row(label, fn):
        return {"label": label, "cells": [{"chips": [], "lists": fn(p)} for p in papers], "shared": []}

    methods = chip_row("Methods", "methods")
    for c, p in zip(methods["cells"], papers):
        c["lists"] = [{"label": None, "items": [_clip(x, 260) for x in p.get("method_notes", [])[:2]]}]
    return [
        list_row("Problem", lambda p: [{"label": None, "items": [_clip(x, 260) for x in p.get("problem", [])[:2]]}]),
        methods,
        chip_row("Datasets", "datasets"),
        chip_row("Metrics", "metrics"),
        list_row("Results", lambda p: [{"label": None, "items": [_clip(x, 260) for x in p.get("results", [])[:3]]}]),
        list_row("Limitations", lambda p: [
            {"label": "Stated in the paper", "items": [_clip(x, 240) for x in p.get("limitations_stated", [])[:2]]},
            {"label": "Noticed by the tool", "items": [_clip(x, 240) for x in p.get("limitations_noticed", [])[:3]]}])]


# ---------- literature review ----------
def _join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else ""


def _surname(p):
    a = (p.get("authors") or [""])[0]
    return citations.split_name(a)[0] if a else ""


def _who(p):
    s, n, y = _surname(p), len(p.get("authors") or []), p.get("year") or "n.d."
    if not s:
        return "the authors"
    if n > 2:
        return f"{s} et al. ({y})"
    if n == 2:
        return f"{s} and {citations.split_name(p['authors'][1])[0]} ({y})"
    return f"{s} ({y})"


def _lower(s):
    s = s.strip().rstrip(".")
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


def _metric(m):
    """Readable metric name inside a sentence: accuracy, F1, BLEU, ROUGE-L, BERTScore ..."""
    low = m.strip().lower()
    if low.startswith("f1") or low == "f-1": return "F1"
    if low == "bertscore": return "BERTScore"
    if low.startswith("rouge") or len(low) <= 5: return low.upper()
    return low


def _result_phrase(p):
    m = RESULT_RX.search(" ".join(p.get("results", [])))
    if not m:
        return ""
    name = _metric(m.group(1))
    art = "an" if name in AN or name[0].lower() in "aeiou" else "a"
    return f", reporting {art} {name} of {m.group(2)}{m.group(3)}"


def _study(p, i):
    who = _who(p)
    title = _clip(p.get("title", ""), 110)
    sent = f"In \u201c{title}\u201d, {who}" if title else who
    sent = sent[:1].upper() + sent[1:]
    methods, datasets, metrics = p.get("methods", [])[:3], [d for d in p.get("datasets", []) if len(d) <= 40][:2], p.get("metrics", [])[:3]
    if methods:
        sent += f" {VERBS[i % 3]} {_join(methods)}"
    elif p.get("method_notes"):
        sent += f" describe the approach as follows: \u201c{_clip(p['method_notes'][0], 150)}\u201d"
    else:
        sent += " report their approach"
    if datasets:
        sent += f" on the {datasets[0]} dataset" if len(datasets) == 1 else f" on the {_join(datasets)} datasets"
    if metrics:
        sent += f", evaluating with {_join([_metric(m) for m in metrics])}"
    return sent + _result_phrase(p) + "."


def _shared(papers, key):
    c = Counter()
    for p in papers:
        c.update({x.lower(): x for x in p.get(key, []) if len(x) <= 40}.keys())
    first = {x.lower(): x for p in papers for x in p.get(key, [])}
    return [first[k] for k, n in c.most_common() if n >= 2][:4]


def review(items):
    papers = [_norm(i) for i in items]
    n = len(papers)
    # topic words must come up in at least two papers and must not be metric names
    kw = Counter(k for p in papers for k in set(p.get("keywords", [])[:8]) if k.lower() not in GENERIC_KW and len(k) > 3 and not re.fullmatch(nlp.METRICS, k, re.I))
    topic = _join([k for k, n in kw.most_common(3) if n >= 2])
    paras = [(f"Research on {topic} has attracted growing attention, and the {n} studies reviewed here approach it in different ways."
              if topic else f"This section reviews {n} recent studies and the way each of them approaches the problem.")]
    paras.append(" ".join(_study(p, i) for i, p in enumerate(papers)))

    parts = []
    for key, one, many in (("methods", "appears in more than one of the studies", "appear in more than one of the studies"),
                           ("datasets", "is a shared benchmark", "are shared benchmarks"),
                           ("metrics", "is a common evaluation measure", "are common evaluation measures")):
        sh = _shared(papers, key)
        if key == "metrics": sh = [_metric(x) for x in sh]
        if sh:
            parts.append(f"{_join(sh)} {one if len(sh) == 1 else many}")
    if parts:
        syn = "Across the studies, " + "; ".join(parts) + ". Differences in the remaining methods and datasets mean the reported results should be compared with care."
    else:
        syn = "The studies differ in the methods, datasets and metrics they use, which makes a direct comparison of their reported results difficult."
    paras.append(syn)

    lim = Counter(x for p in papers for x in p.get("limitations_noticed", []))
    top = [x for x, _ in lim.most_common(3)]
    if top:
        paras.append("Despite these contributions, the studies leave some gaps: " + "; ".join(_lower(x) for x in top)
                     + ". Future work that addresses these gaps would make the findings easier to trust, reproduce and compare.")

    refs = sorted((citations.apa({"title": p.get("title", ""), "year": p.get("year") or None, "authors": p.get("authors", []),
                                  "url": "", "source": ""}) for p in papers), key=str.lower)
    return {"text": "\n\n".join(paras), "references": refs, "paper_count": n,
            "note": "Automatic draft built from extracted facts. Verify each statement against the papers and rewrite it in your own words."}
