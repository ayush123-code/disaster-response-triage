"""Streamlit demo: disaster-response triage on the 2018 Kerala floods."""
from __future__ import annotations

import html as html_lib
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.priority import LEVELS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

PAGES = [
    "Live command centre",
    "Why sentiment isn't enough",
    "Try it yourself",
    "Results",
]
PRIORITY_COLOR = {
    "Critical": "#e5484d",
    "High": "#f76b15",
    "Medium": "#ffc53d",
    "Low": "#8b8d97",
}
EMO_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
PHONE_RE = re.compile(r"\d(?:[\s-]*\d){7,}")
EXAMPLES = [
    "Family of 5 including a baby trapped on the roof in Pandanad, no food or water since yesterday. Please send help",
    "Praying for everyone in Kerala, so heartbreaking",
    "Bridge collapsed near Kochi, road closed",
]


@st.cache_data
def load_kerala() -> pd.DataFrame:
    return pd.read_parquet(RESULTS / "kerala_scored.parquet")


@st.cache_data
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_pipeline():
    from core.pipeline import TriagePipeline

    return TriagePipeline()


def mask_phones(text: str) -> str:
    return PHONE_RE.sub("[phone hidden]", text)


def safe_text(text) -> str:
    return html_lib.escape(mask_phones(html_lib.unescape(str(text))))


def _dark_ax(fig, ax):
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#c4c4c4", labelsize=8)
    ax.xaxis.label.set_color("#c4c4c4")
    ax.yaxis.label.set_color("#c4c4c4")
    for spine in ax.spines.values():
        spine.set_color("#333333")


def badge_html(level: str) -> str:
    color = PRIORITY_COLOR.get(level, "#8b8d97")
    fg = "#111111" if level == "Medium" else "#ffffff"
    return (
        f'<span style="display:inline-block;background:{color};color:{fg};'
        f'padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700;'
        f'letter-spacing:0.02em">{html_lib.escape(str(level))}</span>'
    )


def post_card(row, *, compact: bool = False, dim: bool = False) -> str:
    level = str(row["priority"])
    color = PRIORITY_COLOR.get(level, "#8b8d97")
    opacity = "0.42" if dim else "1"
    text = safe_text(row["text"])
    cat = html_lib.escape(str(row["category_display"]))
    conf = float(row["category_conf"])
    distress = float(row["distress"])
    reason = html_lib.escape(str(row["reason"]))
    meta = (
        f'<div style="font-size:0.78rem;color:#9aa0aa;margin-top:4px">'
        f"{cat} · {conf:.0%} conf · distress {distress:.2f}</div>"
        f'<div style="font-size:0.72rem;color:#6f7480;margin-top:4px">{reason}</div>'
    )
    body = f'<div style="font-size:0.86rem;margin-top:6px;line-height:1.35">{text}</div>'
    if compact:
        extra = ""
    else:
        extra = meta
    return (
        f'<div style="background:#16181d;border:1px solid #2a2e38;border-left:4px solid {color};'
        f'border-radius:8px;padding:10px 12px;margin-bottom:8px;opacity:{opacity};'
        f'word-wrap:break-word;overflow-wrap:anywhere">{badge_html(level)}{body}{extra}</div>'
    )


def needs_help_marker(flag: bool) -> str:
    if flag:
        return '<span style="color:#e5484d;font-weight:700">● needs help</span>'
    return '<span style="color:#6b7280">○ no action needed</span>'


def show_category_chart(seen: pd.DataFrame) -> None:
    counts = seen["category_display"].value_counts().sort_values()
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(1.8, 0.34 * len(counts) + 0.7)))
    _dark_ax(fig, ax)
    ax.barh(counts.index, counts.values, color="#f76b15", height=0.62)
    ax.set_xlabel("Posts so far")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def show_emotion_chart(row) -> None:
    values = [float(row[f"emo_{name}"]) for name in EMO_LABELS]
    colors = ["#e5484d" if name in ("fear", "sadness") else "#6b7280" for name in EMO_LABELS]
    fig, ax = plt.subplots(figsize=(6.2, 2.15))
    _dark_ax(fig, ax)
    ax.barh(EMO_LABELS[::-1], values[::-1], color=colors[::-1], height=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def init_replay_state() -> None:
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("pos", 0)


def page_live(df: pd.DataFrame) -> None:
    init_replay_state()
    ss = st.session_state

    with st.sidebar:
        st.subheader("Replay")
        c1, c2, c3 = st.columns(3)
        if c1.button("Start", use_container_width=True, key="btn_start"):
            ss.running = True
        if c2.button("Pause", use_container_width=True, key="btn_pause"):
            ss.running = False
        if c3.button("Reset", use_container_width=True, key="btn_reset"):
            ss.running = False
            ss.pos = 0
        speed = st.slider("Posts per second", 1, 20, 5, key="replay_speed")

    n = len(df)
    if ss.running and ss.pos < n:
        ss.pos = min(int(ss.pos) + int(speed), n)

    pos = int(ss.pos)
    seen = df.iloc[:pos]

    st.caption(
        "Replaying 1,582 tweets from the 2018 Kerala floods, in posting order. "
        "The queue on the right is what responders should look at first — not the live feed."
    )
    if ss.running:
        st.caption("Status: **live**")
    elif pos >= n and n > 0:
        st.caption("Status: replay finished")
    else:
        st.caption("Status: paused")

    m1, m2, m3, m4, m5 = st.columns(5)
    counts = seen["priority"].value_counts() if not seen.empty else pd.Series(dtype=int)
    m1.metric("Posts processed", pos)
    m2.metric("Critical", int(counts.get("Critical", 0)))
    m3.metric("High", int(counts.get("High", 0)))
    m4.metric("Medium", int(counts.get("Medium", 0)))
    m5.metric("Low (filtered out)", int(counts.get("Low", 0)))

    feed_col, queue_col = st.columns(2)
    with feed_col:
        st.subheader("Incoming feed")
        if seen.empty:
            st.info("Press Start to begin the replay.")
        else:
            latest = seen.iloc[::-1].head(8)
            cards = [
                post_card(row, compact=True, dim=str(row["priority"]) == "Low")
                for _, row in latest.iterrows()
            ]
            st.markdown("".join(cards), unsafe_allow_html=True)

    with queue_col:
        st.subheader("Priority queue")
        if seen.empty:
            st.info("Urgent posts will appear here as they arrive.")
        else:
            queue = (
                seen[seen["priority"] != "Low"]
                .sort_values(["priority_rank", "score"], ascending=[True, False])
                .head(12)
            )
            if queue.empty:
                st.caption("No Critical / High / Medium posts yet — Low posts are filtered out.")
            else:
                cards = [post_card(row) for _, row in queue.iterrows()]
                st.markdown("".join(cards), unsafe_allow_html=True)

    st.subheader("Predicted categories so far")
    if seen.empty:
        st.caption("Counts appear once posts start arriving.")
    else:
        show_category_chart(seen)

    if ss.running:
        if pos >= n:
            ss.running = False
        else:
            time.sleep(1)
            st.rerun()


def comparison_column(title: str, subset: pd.DataFrame) -> None:
    n_help = int(subset["needs_help"].sum())
    st.markdown(f"**{title}**")
    st.markdown(f"{n_help} of {len(subset)} actually need help")
    blocks = []
    for _, row in subset.iterrows():
        text = safe_text(row["text"])
        marker = needs_help_marker(bool(row["needs_help"]))
        blocks.append(
            f'<div style="background:#16181d;border:1px solid #2a2e38;border-radius:8px;'
            f'padding:10px 12px;margin-bottom:8px;word-wrap:break-word;overflow-wrap:anywhere">'
            f'<div style="font-size:0.86rem;line-height:1.35;margin-bottom:6px">{text}</div>'
            f'<div style="font-size:0.78rem">{marker}</div></div>'
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def page_sentiment(df: pd.DataFrame, ranking: dict) -> None:
    st.caption(
        "Prayers and sympathy posts are often very negative, but they need no action. "
        "A calm rescue request can look less 'negative' and still be critical. "
        "Plain sentiment analysis is not enough."
    )
    by_neg = df.sort_values("negativity", ascending=False).head(10)
    by_ours = df.sort_values(["priority_rank", "score"], ascending=[True, False]).head(10)

    left, right = st.columns(2)
    with left:
        comparison_column("Top 10 by plain negative sentiment", by_neg)
    with right:
        comparison_column("Top 10 by our priority", by_ours)

    st.subheader("How many posts a responder must read")
    curve = RESULTS / "ranking_curve.png"
    if curve.exists():
        st.image(str(curve), caption="Posts read vs people found", width="stretch")
    else:
        st.info("Ranking curve image not found.")

    rows = []
    for name, metrics in ranking["ranking"].items():
        rows.append(
            {
                "Ordering": name,
                "Precision@20": metrics["precision_at_20"],
                "Precision@50": metrics["precision_at_50"],
                "Precision@100": metrics["precision_at_100"],
                "Posts read to find 50%": metrics["posts_read_for_50pct"],
                "Posts read to find 90%": metrics["posts_read_for_90pct"],
            }
        )
    table = pd.DataFrame(rows).set_index("Ordering")
    display = table.copy()
    for col in ("Precision@20", "Precision@50", "Precision@100"):
        display[col] = display[col].map(lambda x: f"{x:.0%}")
    st.dataframe(display, width="stretch")
    st.caption(
        "Ground truth comes from human annotators; these tweets were never seen in training."
    )


def _fill_example(text: str) -> None:
    st.session_state.try_text = text


def page_try() -> None:
    st.caption("Paste one post per line, or load an example. First analysis loads the models (~30–60s).")
    b1, b2, b3 = st.columns(3)
    b1.button("Rescue request", on_click=_fill_example, args=(EXAMPLES[0],), use_container_width=True)
    b2.button("Prayer / sympathy", on_click=_fill_example, args=(EXAMPLES[1],), use_container_width=True)
    b3.button("Bridge collapsed", on_click=_fill_example, args=(EXAMPLES[2],), use_container_width=True)

    st.text_area("Posts (one per line)", key="try_text", height=140)
    run = st.button("Analyse", type="primary")
    if not run:
        return

    lines = [ln.strip() for ln in st.session_state.get("try_text", "").splitlines() if ln.strip()]
    if not lines:
        st.warning("Enter at least one post.")
        return

    with st.spinner("Loading models (first time ~30–60s) and analysing…"):
        pipe = get_pipeline()
        out = pipe.analyze(lines)

    st.caption(f"Category model: {pipe.categories.name}")

    if len(out) > 1:
        st.subheader("Ranked by priority")
        ranked = out.sort_values(["priority_rank", "score"], ascending=[True, False])
        st.markdown(
            "".join(post_card(row) for _, row in ranked.iterrows()),
            unsafe_allow_html=True,
        )
        st.subheader("Each post")

    for _, row in out.iterrows():
        st.markdown(post_card(row), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            distress = min(max(float(row["distress"]), 0.0), 1.0)
            st.progress(distress, text=f"Distress {float(row['distress']):.2f}")
            st.write(f"Plain negativity: **{float(row['negativity']):.2f}**")
        with c2:
            show_emotion_chart(row)


def _fmt_metric(value) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.3f}"


def page_results(ranking: dict) -> None:
    st.caption("Classifier quality on the 15,160-tweet HumAID test set, plus Kerala replay stats.")

    baseline_path = RESULTS / "baseline_metrics.json"
    distil_path = RESULTS / "distilbert_metrics.json"
    rows = []
    if baseline_path.exists():
        base = load_json(str(baseline_path))
        rows.append(
            {
                "Model": base.get("model", "TF-IDF + logistic regression"),
                "Accuracy": _fmt_metric(base["accuracy"]),
                "Macro F1": _fmt_metric(base["macro_f1"]),
                "Weighted F1": _fmt_metric(base["weighted_f1"]),
            }
        )
    if distil_path.exists():
        distil = load_json(str(distil_path))
        rows.append(
            {
                "Model": distil.get("model", "DistilBERT (fine-tuned)"),
                "Accuracy": _fmt_metric(distil["accuracy"]),
                "Macro F1": _fmt_metric(distil["macro_f1"]),
                "Weighted F1": _fmt_metric(distil["weighted_f1"]),
            }
        )
    else:
        rows.append(
            {
                "Model": "DistilBERT (fine-tuned)",
                "Accuracy": "training in progress",
                "Macro F1": "training in progress",
                "Weighted F1": "training in progress",
            }
        )
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Confusion matrices")
    img_l, img_r = st.columns(2)
    with img_l:
        base_img = RESULTS / "baseline_confusion.png"
        if base_img.exists():
            st.image(str(base_img), caption="Baseline", width="stretch")
        else:
            st.info("Baseline confusion matrix not found.")
    with img_r:
        distil_img = RESULTS / "distilbert_confusion.png"
        if distil_img.exists():
            st.image(str(distil_img), caption="DistilBERT", width="stretch")
        else:
            st.info("DistilBERT confusion matrix: training in progress.")

    st.subheader("Kerala replay")
    k1, k2, k3 = st.columns(3)
    k1.metric("Posts", ranking["posts"])
    k2.metric("Posts needing help", ranking["posts_needing_help"])
    k3.metric("Category model", ranking["category_model"])
    k4, k5 = st.columns(2)
    k4.metric("Kerala accuracy", f"{ranking['kerala_category_accuracy']:.1%}")
    k5.metric("Kerala macro F1", f"{ranking['kerala_category_macro_f1']:.3f}")

    st.subheader("Priority counts")
    counts = ranking.get("priority_counts", {})
    order = [lvl for lvl in LEVELS if lvl in counts] + [k for k in counts if k not in LEVELS]
    st.dataframe(
        pd.DataFrame({"Priority": order, "Posts": [counts[k] for k in order]}),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(
        page_title="Disaster Response Triage",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Disaster Response Triage: finding the people who need help first")

    page = st.sidebar.radio("Navigate", PAGES, index=0)

    kerala = load_kerala()
    ranking = load_json(str(RESULTS / "ranking.json"))

    if page == "Live command centre":
        page_live(kerala)
    elif page == "Why sentiment isn't enough":
        page_sentiment(kerala, ranking)
    elif page == "Try it yourself":
        page_try()
    else:
        page_results(ranking)


main()
