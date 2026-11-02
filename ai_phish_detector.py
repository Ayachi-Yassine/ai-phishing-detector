"""
AI-Phishing Email Detector (3 classes): legitimate | phishing | ai_phishing
Trained on the PhishFuzzer dataset (https://github.com/DataPhish/PhishFuzzer)

  git clone https://github.com/DataPhish/PhishFuzzer.git
  pip install scikit-learn pandas scipy joblib

  python ai_phish_detector.py train  PhishFuzzer      # trains, evaluates, saves model
  python ai_phish_detector.py predict email.txt       # JSON SOC alert for one email (file or raw text)
"""
import re, sys, os, json, joblib, datetime
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

CLASSES = ["legitimate", "phishing", "ai_phishing"]
URGENT = ["urgent", "immediately", "suspended", "verify", "confirm", "expire", "action required", "locked",
          "security alert", "within 24", "click here", "password", "limited time", "unusual activity", "unauthorized"]
SHORTENERS = ["bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rebrand.ly"]
URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.I)
HAND_NAMES = ["n_urls", "ip_url", "shortened_url", "at_in_url", "urgency_words", "exclamation_rate", "upper_ratio",
              "digit_ratio", "type_token_ratio", "avg_sentence_len", "sentence_len_std", "avg_word_len",
              "punct_density", "repeated_words", "double_spaces", "length"]


def handcrafted(t):
    low = t.lower(); urls = URL_RE.findall(t)
    words = re.findall(r"[A-Za-z']+", t); nw = max(len(words), 1)
    sl = [len(s.split()) for s in re.split(r"[.!?]+\s", t) if s.strip()] or [0]
    return [len(urls),
            int(any(re.search(r"https?://\d+\.\d+\.\d+\.\d+", u) for u in urls)),
            int(any(s in u.lower() for u in urls for s in SHORTENERS)),
            int(any("@" in u for u in urls)),
            sum(low.count(k) for k in URGENT), t.count("!") / nw,
            sum(c.isupper() for c in t) / max(len(t), 1), sum(c.isdigit() for c in t) / max(len(t), 1),
            len(set(w.lower() for w in words)) / nw, float(np.mean(sl)), float(np.std(sl)),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            len(re.findall(r"[,;:]", t)) / nw, len(re.findall(r"\b(\w+) \1\b", low)),
            len(re.findall(r"[ ]{2,}", t)), len(t)]


class Detector:
    def __init__(self, use_hand=True, use_char=True):
        self.use_hand, self.use_char = use_hand, use_char
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=60000, sublinear_tf=True)
        self.sc = StandardScaler()
        self.clf = LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced")

    def _mat(self, texts, fit):
        parts = [self.w.fit_transform(texts) if fit else self.w.transform(texts)]
        if self.use_char:
            parts.append(self.c.fit_transform(texts) if fit else self.c.transform(texts))
        if self.use_hand:
            H = np.array([handcrafted(t) for t in texts], dtype=float)
            parts.append(csr_matrix(self.sc.fit_transform(H) if fit else self.sc.transform(H)))
        return hstack(parts).tocsr()

    def fit(self, texts, y):
        self.clf.fit(self._mat(texts, True), y); return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._mat(texts, False))

    def feature_names(self):
        n = list(self.w.get_feature_names_out())
        if self.use_char: n += ["char:" + f for f in self.c.get_feature_names_out()]
        if self.use_hand: n += HAND_NAMES
        return n


def load_phishfuzzer(folder):
    frames = []
    for f in ["PhishFuzzer_emails_original_seed_v1.json", "PhishFuzzer_emails_entity_rephrased_v1.json"]:
        frames.append(pd.DataFrame(json.load(open(os.path.join(folder, f), encoding="utf-8"))))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Type"].isin(["Phishing", "Valid"])]                    # spam dropped (not phishing, not legit)
    df = df[df["Language"].fillna("en") == "en"]
    df["Body"] = df["Body"].fillna(""); df["Subject"] = df["Subject"].fillna("")
    df = df[df["Body"].str.len() > 20]
    llm = df["Created by"].astype(str).str.upper().eq("LLM")
    df["label"] = np.where(df["Type"] == "Valid", "legitimate", np.where(llm, "ai_phishing", "phishing"))
    # NOTE: LLM-written *valid* emails stay in 'legitimate' so the model can't learn "AI-written = phishing".
    df["text"] = df["Subject"] + "\n" + df["Body"] + "\n" + df["URL"].apply(
        lambda u: " ".join(u) if isinstance(u, list) else str(u or ""))
    df["group"] = df["Original_ID"].fillna(df["No."]).astype(int)
    return df[["text", "label", "group"]].drop_duplicates("text").reset_index(drop=True)


def train(folder, out="ai_phish_model.joblib"):
    df = load_phishfuzzer(folder)
    print("Dataset:\n", df["label"].value_counts().to_string(), "\n")
    # Split by seed email so variants of the same email never sit in both train and test (no leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_i, te_i = next(gss.split(df, df["label"], df["group"]))
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    print(f"Train {len(tr)} | Test {len(te)}")

    results = {}
    for name, kw in [("A) word TF-IDF only", dict(use_hand=False, use_char=False)),
                     ("B) word + char TF-IDF", dict(use_hand=False, use_char=True)),
                     ("C) full: word + char + handcrafted", dict(use_hand=True, use_char=True))]:
        m = Detector(**kw).fit(tr["text"].tolist(), tr["label"].values)
        pred = m.clf.classes_[m.predict_proba(te["text"].tolist()).argmax(1)]
        results[name] = (m, pred, f1_score(te["label"], pred, average="macro"))
        print(f"{name:42s} macro-F1 = {results[name][2]:.4f}")

    best = max(results, key=lambda k: results[k][2])   # pick best variant by macro-F1
    model, pred, _ = results[best]
    print(f"\n=== Final model: {best} ===")
    print(classification_report(te["label"], pred, digits=4))
    print("Confusion matrix (rows=true, cols=pred), order:", list(model.clf.classes_))
    print(confusion_matrix(te["label"], pred, labels=model.clf.classes_))
    joblib.dump(model, out); print("\nSaved ->", out)


_MODEL = None
def load_model(path="ai_phish_model.joblib"):
    global _MODEL
    if _MODEL is None:
        # The model was pickled while this file ran as __main__; make the class findable when imported by other scripts
        import __main__
        if not hasattr(__main__, "Detector"): __main__.Detector = Detector
        _MODEL = joblib.load(path)
    return _MODEL


def analyze(text, path="ai_phish_model.joblib"):
    """Classify one email text and return the alert as a dict (used by the CLI and the SOC watcher)."""
    m = load_model(path)
    p = m.predict_proba([text])[0]; classes = list(m.clf.classes_); k = int(p.argmax())
    verdict = classes[k]
    # Explainability: top features pushing toward the predicted class
    X = m._mat([text], False); names = m.feature_names()
    contrib = X.multiply(m.clf.coef_[k]).tocoo()
    top = sorted(zip(contrib.col, contrib.data), key=lambda z: -z[1])[:30]
    reasons = [names[i] for i, _ in top if not names[i].startswith("char:")][:5]
    h = dict(zip(HAND_NAMES, handcrafted(text)))
    flags = []
    if h["shortened_url"]: flags.append("shortened URL")
    if h["ip_url"]: flags.append("IP-address URL")
    if h["urgency_words"] >= 2: flags.append(f"{int(h['urgency_words'])} urgency keywords")
    return {
        "source": "ai-phish-detector",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdict": verdict,
        "confidence": round(float(p[k]), 3),
        "probabilities": {c: round(float(x), 3) for c, x in zip(classes, p)},
        "severity": {"legitimate": "info", "phishing": "high", "ai_phishing": "critical"}[verdict],
        "indicators": flags, "top_terms": reasons,
        "urls": URL_RE.findall(text)[:10],
    }


def predict(inp, path="ai_phish_model.joblib"):
    text = open(inp, encoding="utf-8", errors="ignore").read() if os.path.isfile(inp) else inp
    alert = analyze(text, path)
    print(json.dumps(alert, indent=2)); return alert


if __name__ == "__main__":
    if len(sys.argv) < 3: print(__doc__)
    elif sys.argv[1] == "train": train(sys.argv[2])
    elif sys.argv[1] == "predict": predict(" ".join(sys.argv[2:]))
