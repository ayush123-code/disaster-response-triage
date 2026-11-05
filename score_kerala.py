"""Score the Kerala floods 2018 replay set and compare ranking strategies.

Ground truth "needs help" = human label is a request/urgent need, injured or dead,
or missing or found people. These tweets were never seen during training.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from core.data import RESULTS, load_kerala
from core.labels import URGENT
from core.pipeline import TriagePipeline


def ranking_metrics(needs_help: np.ndarray) -> dict:
    found = np.cumsum(needs_help)
    total = needs_help.sum()
    return {
        **{f"precision_at_{k}": float(needs_help[:k].mean()) for k in (20, 50, 100, 200)},
        "posts_read_for_50pct": int(np.searchsorted(found, 0.5 * total) + 1),
        "posts_read_for_90pct": int(np.searchsorted(found, 0.9 * total) + 1),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    kerala = load_kerala("test")
    pipe = TriagePipeline()
    scored = pipe.analyze(list(kerala.tweet_text))
    scored.insert(0, "tweet_id", kerala.tweet_id)
    scored.insert(1, "true_category", kerala.class_label)
    scored["needs_help"] = scored.true_category.isin(URGENT)
    scored.to_parquet(RESULTS / "kerala_scored.parquet")

    orders = {
        "Chronological (today)": scored.index.to_numpy(),
        "Plain negative sentiment": scored.negativity.to_numpy().argsort()[::-1],
        "Distress only": scored.distress.to_numpy().argsort()[::-1],
        "Our priority": scored.sort_values(["priority_rank", "score"], ascending=[True, False]).index.to_numpy(),
    }
    needs = scored.needs_help.to_numpy()
    ranking = {name: ranking_metrics(needs[idx]) for name, idx in orders.items()}

    report = {
        "category_model": pipe.categories.name,
        "posts": len(scored),
        "posts_needing_help": int(needs.sum()),
        "kerala_category_accuracy": accuracy_score(scored.true_category, scored.category),
        "kerala_category_macro_f1": f1_score(scored.true_category, scored.category, average="macro"),
        "priority_counts": scored.priority.value_counts().to_dict(),
        "ranking": ranking,
    }
    (RESULTS / "ranking.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(needs) + 1)
    for name, idx in orders.items():
        ax.plot(x, np.cumsum(needs[idx]) / needs.sum() * 100, label=name, lw=2.2 if name == "Our priority" else 1.5)
    ax.set_xlabel("Posts read by responders")
    ax.set_ylabel("People needing help found (%)")
    ax.set_title("Kerala floods 2018: how fast each ordering finds people who need help")
    ax.set_xlim(0, len(needs)); ax.set_ylim(0, 101); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "ranking_curve.png", dpi=150)


if __name__ == "__main__":
    main()
