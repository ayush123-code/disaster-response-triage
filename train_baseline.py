"""Baseline category classifier: TF-IDF features + logistic regression."""
import json
import time

import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline

from core.data import MODELS, RESULTS, load_humaid, load_kerala
from core.labels import CATEGORIES, DISPLAY


def main():
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    train, test = load_humaid("train"), load_humaid("test")

    kerala = load_kerala("test")
    overlap = len(set(kerala.text) & set(train.text))
    print(f"Kerala replay tweets also in training text: {overlap} of {len(kerala)}")

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
        LogisticRegression(max_iter=2000, C=5, class_weight="balanced"),
    )
    t0 = time.time()
    model.fit(train.text, train.class_label)
    print(f"Trained in {time.time() - t0:.1f}s")

    pred = model.predict(test.text)
    metrics = {
        "model": "TF-IDF + logistic regression",
        "accuracy": accuracy_score(test.class_label, pred),
        "macro_f1": f1_score(test.class_label, pred, average="macro"),
        "weighted_f1": f1_score(test.class_label, pred, average="weighted"),
        "per_class": classification_report(test.class_label, pred, output_dict=True),
    }
    print(classification_report(test.class_label, pred, digits=3))
    (RESULTS / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, MODELS / "baseline.joblib")

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        test.class_label, pred, labels=CATEGORIES,
        display_labels=[DISPLAY[c] for c in CATEGORIES],
        normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("Baseline (TF-IDF + LR): normalised confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "baseline_confusion.png", dpi=150)


if __name__ == "__main__":
    main()
