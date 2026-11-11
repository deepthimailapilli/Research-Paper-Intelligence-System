"""Multi-source paper search (no API keys): IEEE, Springer, Elsevier, ACM via Crossref (DOI prefix),
plus OpenAlex and arXiv. Free full-text links and missing abstracts come from OpenAlex. Papers that can be analyzed / downloaded are listed first."""
import re, requests, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

ALL = ["ieee", "springer", "elsevier", "acm", "openalex", "arxiv"]
PREFIX = {"ieee": ("IEEE", "10.1109"), "springer": ("Springer", "10.1007"),
          "elsevier": ("Elsevier", "10.1016"), "acm": ("ACM", "10.1145")}
PRIORITY = {"IEEE": 0, "Springer": 1}
_CACHE = {}

def _c(s): return re.sub(r"<[^>]+>|\s+", " ", s or "").strip()

def _abs(inv):
    """OpenAlex stores abstracts as an inverted index; rebuild the text."""
    return " ".join(k for _, k in sorted((p, k) for k, ps in (inv or {}).items() for p in ps))

def crossref(label, prefix, topic, y1, y2, n):
    r = requests.get(f"https://api.crossref.org/prefixes/{prefix}/works", timeout=15, params={
        "query": topic, "rows": n, "sort": "relevance", "select": "title,author,issued,abstract,URL",
        "filter": f"from-pub-date:{y1},until-pub-date:{y2}", "mailto": "research-assistant@example.com"})
    r.raise_for_status()
    return [{"title": _c((w.get("title") or [""])[0]),
             "year": (w.get("issued", {}).get("date-parts") or [[None]])[0][0],
             "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in w.get("author", [])[:10]],
             "abstract": _c(w.get("abstract")), "url": w.get("URL"), "source": label}
            for w in r.json()["message"]["items"]]

def openalex(topic, y1, y2, n):
    r = requests.get("https://api.openalex.org/works", timeout=15, params={
        "search": topic, "per-page": n, "filter": f"publication_year:{y1}-{y2},has_abstract:true"})
    r.raise_for_status()
    return [{"title": w.get("title"), "year": w.get("publication_year"),
             "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:10]],
             "abstract": _abs(w.get("abstract_inverted_index")), "url": w.get("doi") or w.get("id"),
             "source": "OpenAlex", "pdf_url": (w.get("best_oa_location") or {}).get("pdf_url")}
            for w in r.json().get("results", [])]

def arxiv(topic, y1, y2, n):
    r = requests.get("http://export.arxiv.org/api/query", timeout=15,
                     params={"search_query": "all:" + topic, "max_results": n * 3})
    ns, out = {"a": "http://www.w3.org/2005/Atom"}, []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        y, link = int(e.find("a:published", ns).text[:4]), e.find("a:id", ns).text
        if y1 <= y <= y2:
            out.append({"title": _c(e.find("a:title", ns).text), "year": y,
                "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)[:10]],
                "abstract": _c(e.find("a:summary", ns).text), "url": link, "source": "arXiv",
                "pdf_url": link.replace("/abs/", "/pdf/").replace("http://", "https://")})
    return out[:n]

def _run(src, topic, y1, y2, n):
    try:
        if src in PREFIX: return crossref(*PREFIX[src], topic, y1, y2, min(n * 3, 30))
        return {"openalex": openalex, "arxiv": arxiv}[src](topic, y1, y2, n)
    except Exception:
        return []

def _enrich(p):
    """OpenAlex lookup by DOI (no key): finds a free PDF and fills a missing abstract,
    so more papers can be downloaded and analyzed."""
    m = re.search(r"10\.\d{4,9}/[^\s?#]+", p.get("url") or "")
    need_pdf = not p.get("pdf_url") and p.get("source") != "arXiv"
    need_abs = len((p.get("abstract") or "").strip()) < 50
    if not m or not (need_pdf or need_abs): return
    try:
        w = requests.get("https://api.openalex.org/works/doi:" + m.group(0), timeout=8,
                         params={"select": "best_oa_location,locations,abstract_inverted_index"}).json()
        if need_pdf:
            locs = [w.get("best_oa_location") or {}] + (w.get("locations") or [])
            p["pdf_url"] = next((l.get("pdf_url") for l in locs if l and l.get("pdf_url")), None)
        if need_abs:
            p["abstract"] = _abs(w.get("abstract_inverted_index")) or p.get("abstract") or ""
    except Exception:
        pass

def _rank(p):
    """Tier 0: analyzable + downloadable | 1: analyzable (abstract) | 2: downloadable only | 3: link only.
    Within a tier IEEE comes first, then Springer, then the rest."""
    ab, pdf = len((p.get("abstract") or "").strip()) >= 50, bool(p.get("pdf_url"))
    p["can_analyze"] = ab or pdf
    p["tier"] = 0 if ab and pdf else 1 if ab else 2 if pdf else 3
    return (p["tier"], PRIORITY.get(p.get("source"), 2))

def search(topic, y1=2018, y2=2026, n=5, sources=None):
    """n = papers per source. Analyzable + downloadable first, link-only papers last."""
    srcs = [s for s in (sources or ALL) if s in ALL] or ALL
    key = (topic.lower().strip(), y1, y2, n, tuple(srcs))
    if key in _CACHE: return _CACHE[key]
    with ThreadPoolExecutor(len(srcs)) as ex:
        lists = list(ex.map(lambda s: _run(s, topic, y1, y2, n), srcs))
    with ThreadPoolExecutor(24) as ex:
        list(ex.map(_enrich, [p for L in lists for p in L if p.get("title")]))
    seen, out = set(), []
    for L in lists:
        L.sort(key=_rank)
        for p in L[:n]:
            k = re.sub(r"\W+", "", (p.get("title") or "").lower())
            if k and k not in seen: seen.add(k); out.append(p)
    out.sort(key=_rank)
    _CACHE[key] = out
    return out
