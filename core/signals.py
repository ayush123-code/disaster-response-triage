"""Pretrained sentiment and emotion signals (no training needed)."""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class _Classifier:
    def __init__(self, name: str, max_len: int = 64):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.labels = [self.model.config.id2label[i].lower()
                       for i in range(self.model.config.num_labels)]
        self.max_len = max_len

    @torch.no_grad()
    def probs(self, texts, batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(list(texts[i:i + batch_size]), truncation=True,
                           max_length=self.max_len, padding=True, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, -1).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))


class EmotionScorer(_Classifier):
    """Emotions: anger, disgust, fear, joy, neutral, sadness, surprise."""

    def __init__(self):
        super().__init__(EMOTION_MODEL)

    def score(self, texts):
        p = self.probs(texts)
        emotions = [dict(zip(self.labels, row)) for row in p]
        distress = p[:, self.labels.index("fear")] + p[:, self.labels.index("sadness")]
        return distress, emotions


class SentimentScorer(_Classifier):
    """Plain 3-class sentiment; used to show why negativity alone misranks posts."""

    def __init__(self):
        super().__init__(SENTIMENT_MODEL)

    def negativity(self, texts) -> np.ndarray:
        return self.probs(texts)[:, self.labels.index("negative")]
