"""NLP helpers (no model downloads): PDF parsing, sections, TextRank summary,
key information + rule-based entity recognition, extraction, research-gap analysis."""
import io, re, random, zlib
from collections import Counter
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

SECTIONS = ["abstract","introduction","related work","literature review","background","methodology",
 "methods","method","proposed method","approach","dataset","experiments","experimental setup",
 "results","results and discussion","discussion","evaluation","conclusion","conclusions","future work","references"]
DBS = r"\b(?:IEEE Xplore|Scopus|Web of Science|PubMed|Google Scholar|arXiv|ACM Digital Library|SpringerLink|ScienceDirect|OpenAlex|Semantic Scholar)\b"
MODELS = r"\b(?:BERT|RoBERTa|DistilBERT|ALBERT|GPT-?\d*|T5|BART|LLaMA|Llama ?\d*|Transformers?|LSTM|BiLSTM|GRU|RNN|CNN|ResNet-?\d*|VGG-?\d*|YOLO\w*|GAN|U-Net|ViT|EfficientNet|MobileNet\w*|Random Forest|SVM|Support Vector Machines?|XGBoost|LightGBM|Naive Bayes|KNN|Logistic Regression|Decision Trees?|Gradient Boosting|Word2Vec|GloVe|Autoencoders?|TF-IDF)\b"
DATASETS = r"\b(?:ImageNet|MNIST|CIFAR-?\d*|COCO|SQuAD|GLUE|IMDB|Kaggle|UCI|MIMIC-?\w*|Wikipedia|Common Crawl|CoNLL-?\d*|WikiText|Penn Treebank|SST-?\d?|Yelp|Twitter|Reddit)\b"
METRICS = r"\b(?:accuracy|precision|recall|f1(?:-?score)?|f-?measure|auc|roc|rmse|mae|mse|r2|bleu|rouge(?:-[l\d])?|meteor|bertscore|perplexity|mean average precision|iou|dice|specificity|sensitivity|mcc|kappa|wer|cer|latency|throughput)\b"
ORGS = r"\b(?:Google(?: Research| DeepMind)?|DeepMind|Microsoft(?: Research)?|OpenAI|Meta AI|Facebook|Amazon|IBM|Intel|NVIDIA|Baidu|Alibaba|Tencent|Huawei|Samsung|Anthropic|Stanford|MIT|Harvard|Oxford|Cambridge|Berkeley|Carnegie Mellon|NASA|WHO|IEEE|ACM|Springer|Elsevier|Wiley|NIH|CERN|UNESCO)\b"
ORG_PAT = r"\b(?:[A-Z][A-Za-z&.-]+ ){1,4}(?:University|Institute|College|Laboratory|Laboratories|Corporation|Hospital|Academy|Inc\.|Ltd\.)|\b(?:University|Institute|College|School|Department) of [A-Z][\w-]+(?: (?:and )?[A-Z][\w-]+){0,3}"
LOCS = r"\b(?:India|China|United States|USA|United Kingdom|UK|Germany|France|Japan|Canada|Australia|Brazil|Italy|Spain|Russia|South Korea|Singapore|Pakistan|Bangladesh|Saudi Arabia|UAE|Iran|Turkey|Egypt|Nigeria|South Africa|Mexico|Indonesia|Malaysia|Vietnam|Thailand|Netherlands|Sweden|Switzerland|Europe|Asia|Africa|Beijing|Shanghai|Tokyo|Seoul|London|Paris|Berlin|New York|California|Boston|Delhi|Mumbai|Bangalore|Bengaluru|Hyderabad|Chennai|Kolkata|Pune|Dubai|Toronto|Sydney|Hong Kong)\b"
PRODUCTS = r"\b(?:TensorFlow|PyTorch|Keras|scikit-learn|Scikit-learn|Hugging Face|MATLAB|Python|NLTK|spaCy|OpenCV|Pandas|NumPy|Weka|SPSS|Excel|Tableau|Power BI|Arduino|Raspberry Pi|Android|iOS|Windows|Linux|Docker|Kubernetes|ChatGPT|Gemini|Claude|Copilot|Alexa|Siri|AWS|Azure|Jupyter|Colab|Spark|Hadoop|MongoDB|MySQL|Flask|Django|CUDA|LangChain|FAISS|Elasticsearch)\b"
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_RX = r"\b(?:\d{1,2}\s+%s\.?,?\s+(?:19|20)\d{2}|%s\.?\s+\d{1,2},?\s+(?:19|20)\d{2}|%s\.?\s+(?:19|20)\d{2})\b" % (MONTHS, MONTHS, MONTHS)
NAME = r"\b(?:[A-Z][a-z]+|[A-Z]\.)(?:\s(?:[A-Z]\.|[A-Z][a-z'’-]+)){1,3}\b"
NOTNAME = r"University|Institute|Department|College|School|Engineering|Science|Technology|Computer|Abstract|Keywords|Index|Terms|Journal|International|Conference|Research|Laboratory|Received|Accepted|Published|Deep|Learning|Machine|Neural|Network|Analysis|Approach|Based|Using|Review"
NOTCITE = {"January","February","March","April","May","June","July","August","September","October","November","December","Fig","Table","Section","Accessed","Received","Accepted","Published","Since","Until","From","Before","After","During","Spring","Summer","Winter","Autumn","Version","Release"}

# ---------- basics ----------
def extract_text(data: bytes):
    r = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in r.pages), len(r.pages)

def sentences(t):
    t = re.sub(r"\s+", " ", t)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t) if 40 < len(s) < 500]

def split_sections(text):
    pat = re.compile(r"^\s*(?:\d+\.?\s*|[IVX]+\.\s*)?(%s)\s*$" % "|".join(SECTIONS), re.I | re.M)
    ms, out = list(pat.finditer(text)), {}
    for i, m in enumerate(ms):
        name = m.group(1).title()
        body = text[m.end(): ms[i+1].start() if i+1 < len(ms) else len(text)].strip()
        if name != "References" and len(body) > 200 and name not in out:
            out[name] = body
    return out or {"Full Text": text}

def summarize(t, n=4):
    s = sentences(t)
    if len(s) <= n: return " ".join(s)
    try:
        X = TfidfVectorizer(stop_words="english").fit_transform(s)
        score = (X @ X.T).toarray().sum(1)
        return " ".join(s[i] for i in sorted(np.argsort(-score)[:n]))
    except ValueError:
        return " ".join(s[:n])

def keywords(t, n=10):
    try:
        v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
        X = v.fit_transform(sentences(t) or [t])
        sc = np.asarray(X.sum(0)).ravel()
        return [v.get_feature_names_out()[i] for i in np.argsort(-sc)[:n]]
    except ValueError:
        return []

def pick(sents, rx, n=3):
    return list(dict.fromkeys(s for s in sents if re.search(rx, s, re.I)))[:n]

def retrieve(question, text, k=3):
    s = sentences(text)
    if not s: return []
    try:
        v = TfidfVectorizer(stop_words="english").fit(s + [question])
        sim = (v.transform(s) @ v.transform([question]).T).toarray().ravel()
        return [s[i] for i in np.argsort(-sim)[:k] if sim[i] > 0]
    except ValueError:
        return []

# ---------- key information ----------
def _uniq(rx, text, flags=0):
    seen = {}
    for m in re.findall(rx, text, flags):
        seen.setdefault(re.sub(r"\s+", " ", m).lower(), re.sub(r"\s+", " ", m).strip())
    return list(seen.values())

def _counted(rx, text, flags=0, fmt=lambda x: x):
    cnt, first = Counter(), {}
    for m in re.findall(rx, text, flags):
        m = re.sub(r"\s+", " ", m).strip(); k = m.lower()
        cnt[k] += 1; first.setdefault(k, m)
    return [f"{fmt(first[k])} ×{n}" for k, n in cnt.most_common()]

def _authors(lines):
    names = []
    for l in lines[1:14]:
        if re.match(r"abstract|keywords|index terms|1\.?\s*introduction", l, re.I): break
        for m in re.findall(NAME, re.sub(r"[\d*†‡§¹²³]+", " ", l)):
            if not re.search(NOTNAME, m) and m not in names: names.append(m)
    return names[:12]

def _dates(head):
    lab = [re.sub(r"\s+", " ", l.strip())[:120] for l in head.splitlines()
           if re.search(r"received|accepted|published|revised|submitted|available online", l, re.I)
           and re.search(r"(?:19|20)\d{2}", l)]
    return list(dict.fromkeys(lab))[:4] or _uniq(DATE_RX, head, re.I)[:6]

def _orgs(text):
    found = _uniq(ORGS, text) + _uniq(ORG_PAT, text)
    return list(dict.fromkeys(o for o in found if not o.startswith("The ")))[:12]

def key_info(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sents, authors = sentences(text), _authors(lines)
    cited = Counter(n for n in re.findall(
        r"\b([A-Z][a-z]{2,})(?: et al\.?,? ?\(?| and [A-Z][a-z]+ \(| \()(?:19|20)\d{2}", text) if n not in NOTCITE)
    return {
        "Basics": {"Title": lines[0] if lines else "", "Authors": authors,
                   "Important dates": _dates(text[:4000]), "Keywords": keywords(text, 8)},
        "Models, data & metrics": {
            "Models / methods": _counted(MODELS, text)[:15],
            "Datasets": _counted(DATASETS, text) or pick(sents, r"\bdata ?sets?\b", 2),
            "Databases": _counted(DBS, text, re.I),
            "Metrics": _counted(METRICS, text, re.I, lambda x: x.upper() if len(x) <= 5 else x.capitalize())},
        "Named entities": {
            "People": authors + [f"{n} (cited)" for n, _ in cited.most_common(8)],
            "Organizations": _orgs(text), "Locations": _counted(LOCS, text),
            "Products": _counted(PRODUCTS, text), "Dates": _uniq(DATE_RX, text, re.I)[:10]}}

# ---------- extraction + research gap ----------
def _sec(secs, *names):
    return " ".join(v for k, v in secs.items() if k.lower() in names)

def extraction(text, secs):
    sents = sentences(text)
    intro = sentences(_sec(secs, "abstract", "introduction")) or sents
    meth = sentences(_sec(secs, "methodology", "methods", "method", "proposed method", "approach")) or sents
    res = sentences(_sec(secs, "results", "results and discussion", "evaluation", "experiments")) or sents
    return {
        "Problem": pick(intro, r"problem|challenge|aim|objective|motivat|address", 3) or intro[:2],
        "Methods used": pick(meth, r"we propose|propos|we use|used|algorithm|model|framework|approach", 3),
        "Results": pick(res, r"accuracy|\d+(\.\d+)?\s?%|outperform|f1|precision|recall|improv", 3),
        "Conclusion": sentences(_sec(secs, "conclusion", "conclusions"))[:3] or pick(sents, r"in conclusion|we conclude|concluded", 2)}

# (regex showing the paper DOES this, limitation if absent, suggestion)
CHECKS = [
 (r"cross-?validation|k-fold|p-?value|significan|confidence interval", "No cross-validation or statistical significance testing is evident.", "Add k-fold cross-validation and significance tests (t-test, Wilcoxon) to show gains are not by chance."),
 (r"ablation", "No ablation study is evident, so each component's contribution is unclear.", "Run ablation studies to measure what each component contributes."),
 (r"baseline|state-of-the-art|compared (?:with|to)|comparison", "Comparison with baselines or recent state-of-the-art work looks limited.", "Benchmark against recent state-of-the-art methods on identical data splits."),
 (r"real-?world|real-?time|deploy|production|case study", "No real-world or deployment evaluation is evident.", "Validate in a real-world setting with a pilot deployment, case study or user study."),
 (r"interpret|explainab|shap|lime|attention map|saliency", "Interpretability of the results is not discussed.", "Add explainability analysis (SHAP, LIME, attention visualisation) to build trust."),
 (r"runtime|training time|computational (?:cost|complexity)|latency|inference time|memory", "Computational cost and efficiency are not reported.", "Report training/inference time and memory; try pruning, quantisation or distillation."),
 (r"ethic|bias|fairness|privacy|consent", "Ethical, bias or privacy aspects are not addressed.", "Audit for bias and discuss privacy and ethical implications."),
 (r"github|code is available|open-?source|publicly available|supplementary", "No code or data availability statement, which limits reproducibility.", "Release code, trained models and data splits for reproducibility."),
 (r"hyper-?parameter|grid search|random search|tuning", "Hyperparameter selection is not described.", "Document hyperparameter tuning (grid or Bayesian search) and report sensitivity."),
 (r"error analysis|failure case|misclassif|confusion matrix", "No error analysis of failure cases is evident.", "Add error analysis or a confusion matrix to understand failure cases."),
 (r"multilingual|cross-?domain|generaliz|transfer", "Generalisation across domains, languages or datasets is not tested.", "Test cross-domain and multilingual generalisation, or apply transfer learning.")]
STATED = [(r"dataset|data size|small sample", "Use larger and more diverse datasets with cross-dataset validation."),
          (r"computation|cost|time|complex", "Reduce cost with lighter models, pruning or distillation."),
          (r"bias", "Add fairness-aware evaluation and bias mitigation."),
          (r"noise|noisy|imbalance", "Handle noise and class imbalance with augmentation or re-sampling.")]
GENERIC = ["Combine the approach with recent transformer or foundation-model methods.",
           "Extend to multimodal data (text with images or audio) where relevant.",
           "Study scalability on much larger data volumes.",
           "Reduce labelling effort with semi-supervised or few-shot learning.",
           "Run an expert or user evaluation alongside automatic metrics."]

def _mix(tips, text, n=10):
    """Shuffle suggestions (same paper -> same order) so they do not mirror the limitations list;
    the cross-validation tip is moved to the middle instead of leading."""
    kf = next((t for t in tips if "cross-validation" in t), None)
    rest = [t for t in tips if t is not kf]
    random.Random(zlib.crc32(text[:2000].encode("utf-8", "ignore"))).shuffle(rest)
    rest = rest[: n - 1 if kf else n]
    if kf: rest.insert(len(rest) // 2, kf)
    return rest

def gap_analysis(text, secs):
    low, ex = text.lower(), extraction(text, secs)
    stated = pick(sentences(text), r"limitation|limited|lack|fail|not able|drawback|small dataset|future work|however", 4)
    noticed, tips = [], []
    for rx, lim, tip in CHECKS:
        if not re.search(rx, low): noticed.append(lim); tips.append(tip)
    if len(_uniq(DATASETS, text)) <= 1:
        noticed.append("Evaluation appears to rely on one or very few named datasets.")
        tips.append("Evaluate on additional public datasets to prove robustness.")
    blob = " ".join(stated).lower()
    tips += [t for rx, t in STATED if re.search(rx, blob)]
    tips += GENERIC
    tips = _mix(list(dict.fromkeys(tips)), text)
    return {"Problem statement": ex["Problem"], "Methods used": ex["Methods used"],
            "Limitations stated in the paper": stated,
            "Limitations we noticed": noticed[:8],
            "Suggestions to overcome": tips}
