"""Fine-tune DistilBERT on HumAID humanitarian categories.

Runs on CPU (slow but works) or a GPU such as Google Colab's (fast).
Saves a checkpoint every --save-every steps; rerun with --resume to continue.
    python train_distilbert.py --epochs 1
    python train_distilbert.py --epochs 1 --resume
    python train_distilbert.py --benchmark   # time a few batches, then exit
"""
import argparse
import ctypes
import json
import math
import shutil
import sys
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
OUT_DIR = MODELS / "distilbert-humaid"
CKPT_DIR = MODELS / "distilbert-checkpoint"


def keep_awake():
    """Stop Windows from sleeping while this process runs."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def encode(tokenizer, texts, labels, idx):
    enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                    padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor([labels[i] for i in idx])
    return enc


@torch.no_grad()
def predict(model, tokenizer, texts, device, batch_size=128):
    model.eval()
    loader = DataLoader(list(range(len(texts))), batch_size=batch_size,
                        collate_fn=lambda idx: encode(tokenizer, texts, [0] * len(texts), idx))
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out.append(model(**batch).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def save_checkpoint(model, tokenizer, opt, sched, step, best_f1):
    model.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)
    torch.save({"step": step, "best_f1": best_f1, "opt": opt.state_dict(),
                "sched": sched.state_dict()}, CKPT_DIR / "trainer_state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    keep_awake()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| threads:", torch.get_num_threads(), flush=True)

    train, val, test = (load_humaid(s) for s in ("train", "validation", "test"))
    texts, labels = list(train.text), [LABEL2ID[c] for c in train.class_label]
    y_val = [LABEL2ID[c] for c in val.class_label]

    resuming = args.resume and (CKPT_DIR / "trainer_state.pt").exists()
    src = CKPT_DIR if resuming else BASE
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=len(CATEGORIES), id2label=ID2LABEL, label2id=LABEL2ID).to(device)

    n_batches = math.ceil(len(texts) / args.batch_size)
    steps = math.ceil(n_batches * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    step, best_f1 = 0, -1.0
    if resuming:
        state = torch.load(CKPT_DIR / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, best_f1 = state["step"], state["best_f1"]
        print(f"resumed from step {step}/{steps}", flush=True)

    start_step, t0 = step, time.time()
    while step < steps:
        epoch, offset = divmod(step, n_batches)
        # Same shuffle order for a given epoch, so a resumed run skips exactly the batches already done.
        perm = torch.randperm(len(texts), generator=torch.Generator().manual_seed(epoch)).tolist()
        model.train()
        for b in range(offset, n_batches):
            batch = encode(tok, texts, labels, perm[b * args.batch_size:(b + 1) * args.batch_size])
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1

            done = step - start_step
            if step % 50 == 0 or (args.benchmark and done % 5 == 0):
                rate = (time.time() - t0) / done
                print(f"step {step}/{steps} loss {loss.item():.3f} | {rate:.2f}s/step "
                      f"| ETA {(steps - step) * rate / 60:.0f} min", flush=True)
            if args.benchmark and done == 20:
                return
            if step % args.save_every == 0:
                save_checkpoint(model, tok, opt, sched, step, best_f1)
            if b == n_batches - 1 or step == steps:
                f1 = f1_score(y_val, predict(model, tok, list(val.text), device), average="macro")
                print(f"validation macro F1 at step {step}: {f1:.4f}", flush=True)
                if f1 > best_f1:
                    best_f1 = f1
                    model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)
                save_checkpoint(model, tok, opt, sched, step, best_f1)
                model.train()
            if step >= steps:
                break

    model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
    pred = [ID2LABEL[i] for i in predict(model, tok, list(test.text), device)]
    metrics = {
        "model": "DistilBERT (fine-tuned)",
        "accuracy": accuracy_score(test.class_label, pred),
        "macro_f1": f1_score(test.class_label, pred, average="macro"),
        "weighted_f1": f1_score(test.class_label, pred, average="weighted"),
        "best_val_macro_f1": best_f1,
        "epochs": args.epochs,
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
    shutil.rmtree(CKPT_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
