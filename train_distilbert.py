"""Fine-tune DistilBERT on HumAID humanitarian categories.

Runs on CPU (slow but works) or a GPU such as Google Colab's (fast).
    python train_distilbert.py --epochs 2
    python train_distilbert.py --benchmark   # time a few batches, then exit
"""
import argparse
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from core.data import MODELS, RESULTS, load_humaid
from core.labels import CATEGORIES, DISPLAY, ID2LABEL, LABEL2ID

BASE = "distilbert-base-uncased"
MAX_LEN = 64


def batches(tokenizer, texts, labels, batch_size, shuffle):
    def collate(idx):
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in idx])
        return enc
    return DataLoader(list(range(len(texts))), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out.append(model(**batch).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| threads:", torch.get_num_threads())

    train, val, test = (load_humaid(s) for s in ("train", "validation", "test"))
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(CATEGORIES), id2label=ID2LABEL, label2id=LABEL2ID).to(device)

    y = lambda df: [LABEL2ID[c] for c in df.class_label]
    train_dl = batches(tok, list(train.text), y(train), args.batch_size, True)
    val_dl = batches(tok, list(val.text), y(val), 128, False)
    test_dl = batches(tok, list(test.text), y(test), 128, False)

    steps = math.ceil(len(train_dl) * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    step, best_f1, t0 = 0, -1.0, time.time()
    out_dir = MODELS / "distilbert-humaid"
    while step < steps:
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0 or (args.benchmark and step % 5 == 0):
                rate = (time.time() - t0) / step
                print(f"step {step}/{steps} loss {loss.item():.3f} | {rate:.2f}s/step "
                      f"| ETA {(steps - step) * rate / 60:.0f} min", flush=True)
            if args.benchmark and step == 20:
                return
            if step % len(train_dl) == 0 or step == steps:
                f1 = f1_score(y(val), predict(model, val_dl, device), average="macro")
                print(f"validation macro F1 at step {step}: {f1:.4f}", flush=True)
                if f1 > best_f1:
                    best_f1 = f1
                    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
                model.train()
            if step >= steps:
                break

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    pred = [ID2LABEL[i] for i in predict(model, test_dl, device)]
    metrics = {
        "model": "DistilBERT (fine-tuned)",
        "accuracy": accuracy_score(test.class_label, pred),
        "macro_f1": f1_score(test.class_label, pred, average="macro"),
        "weighted_f1": f1_score(test.class_label, pred, average="weighted"),
        "best_val_macro_f1": best_f1,
        "epochs": args.epochs,
        "train_minutes": (time.time() - t0) / 60,
        "per_class": classification_report(test.class_label, pred, output_dict=True),
    }
    print(classification_report(test.class_label, pred, digits=3))
    (RESULTS / "distilbert_metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        test.class_label, pred, labels=CATEGORIES,
        display_labels=[DISPLAY[c] for c in CATEGORIES],
        normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("DistilBERT: normalised confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "distilbert_confusion.png", dpi=150)


if __name__ == "__main__":
    main()
