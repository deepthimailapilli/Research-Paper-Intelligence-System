"""APA references used in generated literature-review drafts."""
import re

PARTICLES = {"van", "von", "de", "der", "den", "da", "del", "della", "di", "du", "la", "le", "bin", "al", "dos", "das"}
SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}
PUBLISHER = {"IEEE": "IEEE", "Springer": "Springer", "Elsevier": "Elsevier", "ACM": "ACM", "arXiv": "arXiv"}


def split_name(name):
    """'Given Middle Family' or 'Family, Given' -> (family, [given names])."""
    name = re.sub(r"\s+", " ", (name or "")).strip()
    if not name:
        return "", []
    if "," in name:
        fam, giv = name.split(",", 1)
        return fam.strip(), giv.split()
    parts = [p for p in name.split() if p.lower() not in SUFFIXES]
    if len(parts) == 1:
        return parts[0], []
    fam = parts[-1]
    i = len(parts) - 1
    while i > 1 and parts[i - 1].lower() in PARTICLES:
        i -= 1
        fam = parts[i] + " " + fam
    return fam, parts[:i]


def _initials(given, with_space=True):
    out = []
    for g in given:
        bits = [b[0].upper() + "." for b in g.split("-") if b and b[0].isalpha()]
        if bits:
            out.append("-".join(bits))
    return (" " if with_space else "").join(out)


def doi_of(url):
    m = re.search(r"10\.\d{4,9}/[^\s?#]+", url or "")
    return m.group(0).rstrip(".,;)") if m else ""


def _title(p):
    return re.sub(r"\s+", " ", (p.get("title") or "")).strip().rstrip(".")


def _publisher(p):
    return PUBLISHER.get(p.get("source") or "", "")


def _link(p):
    d = doi_of(p.get("url"))
    return ("https://doi.org/" + d) if d else (p.get("url") or "")


def _end(s):
    s = s.strip()
    return s if s and s[-1] in ".?!" else s + "."


# ---------- APA 7 ----------
def _apa_authors(authors):
    names = []
    for a in authors:
        fam, giv = split_name(a)
        if fam:
            ini = _initials(giv)
            names.append(f"{fam}, {ini}" if ini else fam)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1] if len(names) > 2 else f"{names[0]}, & {names[1]}"


def apa(p):
    au, yr = _apa_authors(p.get("authors") or []), p.get("year") or "n.d."
    title = _title(p)
    head = f"{_end(au)} ({yr}). " if au else ""
    body = (_end(title) + " ") if au else (_end(title) + f" ({yr}). ")
    pub = _publisher(p)
    tail = " ".join(x for x in [_end(pub) if pub else "", _link(p)] if x)
    return (head + body + tail).strip()

