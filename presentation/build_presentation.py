"""Build the presentation deck from the project's real results.

    .venv\Scripts\python presentation\build_presentation.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
ASSETS = HERE / "assets"

NAVY = RGBColor(0x11, 0x18, 0x27)
RED = RGBColor(0xE5, 0x48, 0x4D)
ORANGE = RGBColor(0xF7, 0x6B, 0x15)
YELLOW = RGBColor(0xE8, 0xA3, 0x17)
TEAL = RGBColor(0x0F, 0x9D, 0x8A)
BLUE = RGBColor(0x2F, 0x6F, 0xD8)
GREY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK = RGBColor(0x1F, 0x29, 0x37)
SOFT = RGBColor(0xC9, 0xD2, 0xE3)
FONT = "Segoe UI"

# Measured with the TF-IDF baseline before DistilBERT replaced it in ranking.json.
BASELINE_KERALA = {"accuracy": 0.740, "macro_f1": 0.584}

base = json.loads((RESULTS / "baseline_metrics.json").read_text())
bert = json.loads((RESULTS / "distilbert_metrics.json").read_text())
rank = json.loads((RESULTS / "ranking.json").read_text())
R = rank["ranking"]
ours, chrono, senti, distress = (R["Our priority"], R["Chronological (today)"],
                                 R["Plain negative sentiment"], R["Distress only"])

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None, dash=False):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.5)
        if dash:
            s.line.dash_style = 4
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, content, size=18, color=DARK, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    """content: a str, or a list of paragraphs, each a str or (str, style dict)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, p in enumerate(content if isinstance(content, list) else [content]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        style = {}
        if isinstance(p, tuple):
            p, style = p
        para.space_after = Pt(style.get("space", 6))
        r = para.add_run()
        r.text = p
        f = r.font
        f.name = FONT
        f.size = Pt(style.get("size", size))
        f.bold = style.get("bold", bold)
        f.italic = style.get("italic", italic)
        f.color.rgb = style.get("color", color)
    return tb


def box_text(shape, lines, size=16, color=WHITE):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        r = para.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size if i == 0 else size - 4)
        r.font.bold = i == 0
        r.font.color.rgb = color


def header(slide, title, number):
    rect(slide, 0, 0, SW, Inches(1.1), NAVY)
    rect(slide, 0, Inches(1.1), SW, Inches(0.07), RED)
    text(slide, Inches(0.6), Inches(0.15), Inches(12), Inches(0.8), title,
         size=30, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, Inches(0.6), Inches(7.0), Inches(9), Inches(0.35),
         "Disaster Response Triage  |  Sentiment analysis for crisis tweets", size=11, color=GREY)
    text(slide, Inches(12.2), Inches(7.0), Inches(0.9), Inches(0.35), str(number),
         size=11, color=GREY, align=PP_ALIGN.RIGHT)


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body


def arrow(slide, x1, y1, x2, y2, color=GREY):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(2.25)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd",
                             {"type": "triangle", "w": "med", "len": "med"}))


def table(slide, x, y, w, rows, col_widths, size=15, highlight_row=None, row_h=0.5):
    t = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h) * len(rows)).table
    for c, cw in enumerate(col_widths):
        t.columns[c].width = Inches(cw)
    for r in range(len(rows)):
        t.rows[r].height = Inches(row_h)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(val)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.name, f.size = FONT, Pt(size)
            f.bold = r == 0 or r == highlight_row
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = NAVY
                f.color.rgb = WHITE
            elif r == highlight_row:
                cell.fill.fore_color.rgb = RGBColor(0xFD, 0xE8, 0xE8)
                f.color.rgb = RED
            else:
                cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
                f.color.rgb = DARK
    return t


def stat(slide, x, y, w, big, label, color):
    rect(slide, x, y, w, Inches(1.55), LIGHT)
    rect(slide, x, y, Inches(0.1), Inches(1.55), color)
    text(slide, x + Inches(0.3), y + Inches(0.12), w - Inches(0.4), Inches(0.8), big,
         size=36, bold=True, color=color)
    text(slide, x + Inches(0.3), y + Inches(0.9), w - Inches(0.4), Inches(0.6), label, size=13)


def f1_chart(path):
    from core.labels import CATEGORIES, DISPLAY
    cats = sorted(CATEGORIES, key=lambda c: bert["per_class"][c]["f1-score"])
    y = range(len(cats))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh([i + 0.2 for i in y], [base["per_class"][c]["f1-score"] for c in cats], 0.4,
            label="Baseline (TF-IDF + LR)", color="#9ca3af")
    ax.barh([i - 0.2 for i in y], [bert["per_class"][c]["f1-score"] for c in cats], 0.4,
            label="DistilBERT", color="#e5484d")
    ax.set_yticks(list(y), [DISPLAY[c] for c in cats], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score per category (HumAID test set)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# 1. Title
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, NAVY)
rect(s, Inches(0.8), Inches(1.45), Inches(0.12), Inches(3.35), RED)
text(s, Inches(1.2), Inches(1.4), Inches(11.5), Inches(0.9),
     "\u201cA family including small kids trapped in the below address plz help them, "
     "water level is increasing.\u201d  (real tweet, Kerala 2018)",
     size=19, color=ORANGE, italic=True)
text(s, Inches(1.2), Inches(2.5), Inches(11.8), Inches(1.0),
     "Finding the People Who Need Help First", size=44, color=WHITE, bold=True)
text(s, Inches(1.2), Inches(3.5), Inches(11.8), Inches(1.2),
     "Disaster response triage with sentiment analysis, tested on real tweets "
     "from the 2018 Kerala floods", size=22, color=SOFT)
text(s, Inches(1.2), Inches(5.4), Inches(11), Inches(1.1), [
    ("Module V: NLP Applications and Ethics", {"size": 17, "color": WHITE, "bold": True}),
    ("Presented by: [Your Name]  |  [Roll No.]  |  [College / Department]", {"size": 15, "color": SOFT}),
])
notes(s, "Open with the real Kerala tweet. During the floods, thousands of people posted for help "
         "on Twitter, buried among prayers, news and donation appeals. Our system finds them first.")

# 2. Problem and key insight
s = prs.slides.add_slide(BLANK)
header(s, "The Problem: Calls for Help Get Buried", 2)
text(s, Inches(0.6), Inches(1.5), Inches(6.2), Inches(3.5), [
    ("\u2022  During disasters, people post for help when phone lines fail", {"size": 19, "space": 12}),
    (f"\u2022  Only {rank['posts_needing_help']} of {rank['posts']:,} Kerala tweets "
     f"({rank['posts_needing_help'] / rank['posts']:.0%}) come from people who need help",
     {"size": 19, "space": 12}),
    ("\u2022  The rest: prayers, news, donation appeals, opinions", {"size": 19, "space": 12}),
    (f"\u2022  Reading in posting order, a responder must read {chrono['posts_read_for_90pct']:,} "
     "posts to find 90% of them", {"size": 19, "space": 12}),
])
rect(s, Inches(7.2), Inches(1.5), Inches(5.5), Inches(5.1), LIGHT)
text(s, Inches(7.5), Inches(1.65), Inches(5), Inches(0.5), "Why plain sentiment analysis fails",
     size=19, bold=True, color=RED)
for i, (quote, verdict, col) in enumerate([
        ("\u201cPraying for everyone in Kerala, so heartbreaking\u201d", "Negativity 0.78  \u2192  no action needed", GREY),
        ("\u201cNeed a boat at Aluva, 3 kids stuck on the roof, water rising fast\u201d", "Negativity 0.78  \u2192  life at risk", RED)]):
    y = Inches(2.35 + i * 1.55)
    rect(s, Inches(7.5), y, Inches(4.9), Inches(1.35), WHITE)
    rect(s, Inches(7.5), y, Inches(0.08), Inches(1.35), col)
    text(s, Inches(7.75), y + Inches(0.1), Inches(4.5), Inches(0.8), quote, size=15, italic=True)
    text(s, Inches(7.75), y + Inches(0.88), Inches(4.5), Inches(0.4), verdict, size=13, bold=True, color=col)
text(s, Inches(7.5), Inches(5.5), Inches(4.9), Inches(1.0),
     "Same sentiment score, opposite urgency. We need what the post is about AND how distressed it sounds.",
     size=14, bold=True, color=NAVY)
notes(s, "Key insight: a standard sentiment model gives a prayer and a rescue request the same negativity. "
         "Sorting by sentiment alone is actually worse than reading in order.")

# 3. Architecture
s = prs.slides.add_slide(BLANK)
header(s, "System Architecture", 3)
bh = Inches(1.3)
y_mid, y_top, y_bot = Inches(2.65), Inches(1.6), Inches(3.7)
boxes = {
    "post": (Inches(0.4), y_mid, Inches(1.8), ["Incoming tweet", "live stream / replay"], NAVY),
    "clean": (Inches(2.6), y_mid, Inches(1.8), ["Clean text", "links, @users, RT"], GREY),
    "cat": (Inches(4.9), y_top, Inches(2.6), ["Category classifier", "fine-tuned DistilBERT, 10 classes"], BLUE),
    "emo": (Inches(4.9), y_bot, Inches(2.6), ["Emotion model", "distress = fear + sadness"], TEAL),
    "prio": (Inches(8.1), y_mid, Inches(2.1), ["Priority rule", "Critical / High / Medium / Low"], RED),
    "queue": (Inches(10.6), y_mid, Inches(2.3), ["Ranked queue", "command-centre app + reason"], ORANGE),
}
for x, y, w, lines, col in boxes.values():
    box_text(rect(s, x, y, w, bh, col, shape=MSO_SHAPE.ROUNDED_RECTANGLE), lines, size=16)


def edge(a, b):
    ax, ay, aw, *_ = boxes[a]
    bx, by, *_ = boxes[b]
    arrow(s, ax + aw + Inches(0.03), ay + bh // 2, bx - Inches(0.03), by + bh // 2)


for a, b in [("post", "clean"), ("clean", "cat"), ("clean", "emo"), ("cat", "prio"), ("emo", "prio"), ("prio", "queue")]:
    edge(a, b)
cmp_box = rect(s, Inches(4.9), Inches(5.15), Inches(2.6), Inches(0.75), None, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
               line=GREY, dash=True)
box_text(cmp_box, ["Plain sentiment model", "comparison only (RoBERTa)"], size=13, color=GREY)

text(s, Inches(8.1), Inches(4.25), Inches(4.8), Inches(0.4), "Priority rule (transparent, every post shows its reason)",
     size=13, bold=True, color=NAVY)
table(s, Inches(8.1), Inches(4.65), Inches(4.8), [
    ["Predicted category", "Distress \u2265 0.6", "< 0.6"],
    ["Request / injured / missing", "Critical", "High"],
    ["Displaced / evacuations", "High", "Medium"],
    ["Damage / caution", "Medium", "Medium"],
    ["Donations, prayers, other", "Low", "Low"],
], [2.6, 1.1, 1.1], size=11, row_h=0.36)
text(s, Inches(0.4), Inches(6.2), Inches(7.4), Inches(0.7),
     "Within a level, posts are sorted by 0.75 \u00d7 P(urgent category) + 0.25 \u00d7 distress.",
     size=13, italic=True, color=GREY)
notes(s, "Two signals per post: what it is about (DistilBERT classifier) and how distressed it sounds "
         "(emotion model). A transparent rule combines them. The plain sentiment model is only used to show "
         "why sentiment alone is not enough.")

# 4. Data and models
s = prs.slides.add_slide(BLANK)
header(s, "Data and Models", 4)
rect(s, Inches(0.6), Inches(1.5), Inches(4.6), Inches(5.2), LIGHT)
rect(s, Inches(0.6), Inches(1.5), Inches(4.6), Inches(0.1), NAVY)
text(s, Inches(0.85), Inches(1.75), Inches(4.2), Inches(4.8), [
    ("HumAID dataset (QCRI)", {"size": 19, "bold": True, "color": NAVY, "space": 10}),
    ("\u2022  Human-labelled tweets from 19 real disasters, 2016\u20132019", {"size": 15, "space": 8}),
    ("\u2022  10 humanitarian categories", {"size": 15, "space": 8}),
    ("\u2022  53,531 train  |  7,793 validation  |  15,160 test", {"size": 15, "space": 18}),
    ("Kerala floods 2018 replay", {"size": 19, "bold": True, "color": NAVY, "space": 10}),
    (f"\u2022  {rank['posts']:,} held-out tweets used for the live demo", {"size": 15, "space": 8}),
    ("\u2022  0 overlap with training text (verified)", {"size": 15, "space": 8}),
    (f"\u2022  {rank['posts_needing_help']} truly need help (human labels)", {"size": 15}),
])
models = [
    ("1. Baseline classifier", "TF-IDF (1\u20132 word n-grams) + logistic regression, class-balanced. Trains in 41 s on a laptop.", GREY),
    ("2. Main classifier", "DistilBERT fine-tuned on HumAID: 2 epochs, max 64 tokens, AdamW, Colab T4 GPU.", BLUE),
    ("3. Distress (sentiment core)", "j-hartmann/emotion-english-distilroberta-base: 7 emotions; distress = fear + sadness.", TEAL),
    ("4. Plain sentiment (comparison)", "cardiffnlp/twitter-roberta-base-sentiment-latest: negativity score.", ORANGE),
]
for i, (title, body, col) in enumerate(models):
    y = Inches(1.5 + i * 1.32)
    rect(s, Inches(5.6), y, Inches(7.1), Inches(1.18), LIGHT)
    rect(s, Inches(5.6), y, Inches(0.1), Inches(1.18), col)
    text(s, Inches(5.9), y + Inches(0.08), Inches(6.7), Inches(0.45), title, size=17, bold=True, color=col)
    text(s, Inches(5.9), y + Inches(0.5), Inches(6.7), Inches(0.65), body, size=13)
notes(s, "Models 1 and 2 answer 'what is the post about'. Model 3 answers 'how distressed is the writer'. "
         "Model 4 is only a comparison point.")

# 5. Classifier accuracy
s = prs.slides.add_slide(BLANK)
header(s, "Classifier Accuracy", 5)
text(s, Inches(0.6), Inches(1.45), Inches(6), Inches(0.4), "HumAID test set (15,160 tweets, 19 disasters)",
     size=15, bold=True, color=NAVY)
table(s, Inches(0.6), Inches(1.9), Inches(6), [
    ["Model", "Accuracy", "Macro F1", "Weighted F1"],
    ["TF-IDF + logistic regression", f"{base['accuracy']:.1%}", f"{base['macro_f1']:.3f}", f"{base['weighted_f1']:.3f}"],
    ["DistilBERT (fine-tuned)", f"{bert['accuracy']:.1%}", f"{bert['macro_f1']:.3f}", f"{bert['weighted_f1']:.3f}"],
], [2.7, 1.1, 1.1, 1.1], size=14, highlight_row=2)
text(s, Inches(0.6), Inches(3.75), Inches(6), Inches(0.4), "Kerala floods replay (unseen disaster)",
     size=15, bold=True, color=NAVY)
table(s, Inches(0.6), Inches(4.2), Inches(6), [
    ["Model", "Accuracy", "Macro F1"],
    ["TF-IDF + logistic regression", f"{BASELINE_KERALA['accuracy']:.1%}", f"{BASELINE_KERALA['macro_f1']:.3f}"],
    ["DistilBERT (fine-tuned)", f"{rank['kerala_category_accuracy']:.1%}", f"{rank['kerala_category_macro_f1']:.3f}"],
], [2.7, 1.65, 1.65], size=14, highlight_row=2)
text(s, Inches(0.6), Inches(6.0), Inches(6), Inches(0.9),
     "Macro F1 weights all 10 categories equally, which matters because classes range from "
     "14,891 donation tweets to 250 missing-people tweets.", size=12, italic=True, color=GREY)
f1_png = ASSETS / "f1_by_category.png"
f1_chart(f1_png)
s.shapes.add_picture(str(f1_png), Inches(6.9), Inches(1.6), width=Inches(6.1))
notes(s, "DistilBERT beats the baseline overall and on the unseen Kerala disaster. The hardest category is "
         "'request / urgent need': requests use very varied language and are often confused with donation drives.")

# 6. Key results
s = prs.slides.add_slide(BLANK)
header(s, "Key Result: Responders Find People Faster", 6)
stat(s, Inches(0.6), Inches(1.5), Inches(4.0), f"{round(ours['precision_at_50'] * 50)} of 50",
     f"top posts need help with our ranking (vs {round(chrono['precision_at_50'] * 50)} in posting order)", RED)
stat(s, Inches(0.6), Inches(3.2), Inches(4.0),
     f"{chrono['posts_read_for_90pct'] / ours['posts_read_for_90pct']:.1f}\u00d7 faster",
     f"to find 90% of people: {ours['posts_read_for_90pct']} posts instead of {chrono['posts_read_for_90pct']:,}", ORANGE)
stat(s, Inches(0.6), Inches(4.9), Inches(4.0), f"{senti['precision_at_50']:.0%}",
     "top-50 precision when sorting by plain sentiment: worse than posting order", GREY)
s.shapes.add_picture(str(RESULTS / "ranking_curve.png"), Inches(5.9), Inches(1.4), width=Inches(6.3))
table(s, Inches(4.9), Inches(5.35), Inches(7.9), [
    ["Ordering", "Top 20", "Top 50", "Top 100", "Posts to find 90%"],
    *[[name, f"{v['precision_at_20']:.0%}", f"{v['precision_at_50']:.0%}", f"{v['precision_at_100']:.0%}",
       f"{v['posts_read_for_90pct']:,}"] for name, v in R.items()],
], [2.9, 1.1, 1.1, 1.1, 1.7], size=11, highlight_row=4, row_h=0.3)
notes(s, "Ground truth is the human labels of 1,582 Kerala tweets never seen in training. "
         "49 of our top 50 posts are real people needing help. Emotion alone is the worst ordering, "
         "because prayers are full of sadness: you need both signals.")

# 7. Live demo
s = prs.slides.add_slide(BLANK)
header(s, "Live Demo: Command Centre", 7)
s.shapes.add_picture(str(ASSETS / "app_live.png"), Inches(0.6), Inches(1.45), height=Inches(5.35))
text(s, Inches(10.3), Inches(1.5), Inches(2.8), Inches(5.3), [
    ("Live command centre", {"size": 16, "bold": True, "color": RED, "space": 4}),
    ("Kerala tweets replayed in posting order; urgent ones rise to the top with a reason.", {"size": 13, "space": 14}),
    ("Why sentiment isn't enough", {"size": 16, "bold": True, "color": RED, "space": 4}),
    ("Top 10 by sentiment vs top 10 by our priority.", {"size": 13, "space": 14}),
    ("Try it yourself", {"size": 16, "bold": True, "color": RED, "space": 4}),
    ("Type any post and see its category, distress and priority.", {"size": 13, "space": 14}),
    ("Privacy", {"size": 16, "bold": True, "color": RED, "space": 4}),
    ("Phone numbers are masked automatically.", {"size": 13}),
])
notes(s, "Switch to the app here. Replay at 10 posts per second, pause and read the top card's reason. "
         "Then type: 'Praying for Kerala, so heartbreaking' (Low) and 'Please help! My grandmother is trapped "
         "on the second floor of our house in Chengannur, she cannot walk and water is rising' (Critical).")

# 8. Limitations and future work
s = prs.slides.add_slide(BLANK)
header(s, "Limitations and Future Improvements", 8)
cols = [
    ("Limitations", RED, [
        "English only: many Kerala posts were in Malayalam or Hinglish",
        "Implicit requests missed: \u201cmy grandmother is stuck, she can't walk\u201d is ranked Low without the words \u201cplease help\u201d",
        "25% of real requests are confused with donation drives",
        "\u201cInjured or dead\u201d includes death-toll news, which ranks high but is not a rescue",
        "Distress saturates: most disaster posts sound fearful, so the category does most of the work",
        "Evaluated on human labels, not on real rescue outcomes",
    ]),
    ("Future improvements", TEAL, [
        "Multilingual model (XLM-R / MuRIL) for Malayalam, Hindi and code-mixed text",
        "Train urgency directly on labelled examples instead of a hand-written rule",
        "Extract locations with NER and show incidents on a live map",
        "Group duplicate reports of the same incident",
        "Separate first-hand reports from news and rumours",
        "Phone SOS page and a live social-media stream",
    ]),
]
for i, (title, col, items) in enumerate(cols):
    x = Inches(0.6 + i * 6.25)
    rect(s, x, Inches(1.5), Inches(6.0), Inches(5.25), LIGHT)
    rect(s, x, Inches(1.5), Inches(6.0), Inches(0.1), col)
    text(s, x + Inches(0.25), Inches(1.7), Inches(5.5), Inches(0.5), title, size=20, bold=True, color=col)
    text(s, x + Inches(0.25), Inches(2.3), Inches(5.5), Inches(4.4),
         [("\u2022  " + it, {"size": 16, "space": 13}) for it in items])
notes(s, "Be upfront: false negatives are the biggest risk in disaster response, so the system only ranks "
         "posts and never hides them. A human always makes the final call.")

# 9. Thank you
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, NAVY)
text(s, 0, Inches(2.3), SW, Inches(1.2), "Thank you!  Questions?", size=48, color=WHITE, bold=True,
     align=PP_ALIGN.CENTER)
text(s, 0, Inches(3.6), SW, Inches(0.8),
     f"{round(ours['precision_at_50'] * 50)} of our top 50 posts are people who need help  |  "
     f"{chrono['posts_read_for_90pct'] / ours['posts_read_for_90pct']:.1f}\u00d7 faster than reading in order",
     size=20, color=ORANGE, align=PP_ALIGN.CENTER)
text(s, 0, Inches(4.6), SW, Inches(0.6), "Code: github.com/ayush123-code/disaster-response-triage",
     size=16, color=SOFT, align=PP_ALIGN.CENTER)
text(s, 0, Inches(5.1), SW, Inches(0.6), "Data: HumAID (Alam et al., ICWSM 2021), CrisisNLP / QCRI",
     size=14, color=SOFT, align=PP_ALIGN.CENTER)

out = HERE / "Disaster_Response_Triage.pptx"
prs.save(out)
print("Saved", out, "with", len(prs.slides), "slides")
