"""End-to-end triage: category + distress -> priority, for a batch of posts."""
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.data import MODELS, clean
from core.labels import CATEGORIES, DISPLAY, URGENT
from core.priority import LEVEL_RANK, priority
from core.signals import DEVICE, EmotionScorer, SentimentScorer


class CategoryModel:
    """Fine-tuned DistilBERT if it has been trained, otherwise the TF-IDF baseline."""

    def __init__(self, kind: str = "auto"):
        bert_dir = MODELS / "distilbert-humaid"
        if kind == "auto":
            kind = "distilbert" if (bert_dir / "config.json").exists() else "baseline"
        self.kind = kind
        if kind == "distilbert":
            self.tok = AutoTokenizer.from_pretrained(bert_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(DEVICE).eval()
            self.order = [self.model.config.id2label[i] for i in range(len(CATEGORIES))]
        else:
            self.model = joblib.load(MODELS / "baseline.joblib")
            self.order = list(self.model.classes_)
        self.name = "DistilBERT (fine-tuned)" if kind == "distilbert" else "TF-IDF + logistic regression"

    @torch.no_grad()
    def probs(self, texts, batch_size: int = 64) -> np.ndarray:
        if self.kind == "baseline":
            p = self.model.predict_proba(list(texts))
        else:
            out = []
            for i in range(0, len(texts), batch_size):
                enc = self.tok(list(texts[i:i + batch_size]), truncation=True, max_length=64,
                               padding=True, return_tensors="pt").to(DEVICE)
                out.append(torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
            p = np.concatenate(out)
        return p[:, [self.order.index(c) for c in CATEGORIES]]


class TriagePipeline:
    def __init__(self, category_model: str = "auto", with_sentiment: bool = True):
        self.categories = CategoryModel(category_model)
        self.emotions = EmotionScorer()
        self.sentiment = SentimentScorer() if with_sentiment else None

    def analyze(self, texts) -> pd.DataFrame:
        cleaned = [clean(t) for t in texts]
        p = self.categories.probs(cleaned)
        distress, emotions = self.emotions.score(cleaned)
        negativity = self.sentiment.negativity(cleaned) if self.sentiment else np.full(len(texts), np.nan)
        urgent_idx = [CATEGORIES.index(c) for c in URGENT]

        rows = []
        for i, text in enumerate(texts):
            cat = CATEGORIES[int(p[i].argmax())]
            top_emotion = max(emotions[i], key=emotions[i].get)
            level, score, reason = priority(cat, float(p[i].max()), float(p[i, urgent_idx].sum()),
                                            float(distress[i]), top_emotion)
            rows.append({
                "text": text,
                "category": cat,
                "category_display": DISPLAY[cat],
                "category_conf": float(p[i].max()),
                "p_urgent": float(p[i, urgent_idx].sum()),
                "distress": float(distress[i]),
                "top_emotion": top_emotion,
                **{f"emo_{k}": float(v) for k, v in emotions[i].items()},
                "negativity": float(negativity[i]),
                "priority": level,
                "priority_rank": LEVEL_RANK[level],
                "score": score,
                "reason": reason,
            })
        return pd.DataFrame(rows)
