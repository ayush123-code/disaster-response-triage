import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

_URL = re.compile(r"https?://\S+")
_USER = re.compile(r"@\w+")
_SPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    text = _URL.sub("http", text)
    text = _USER.sub("@user", text)
    text = text.replace("RT @user:", "").replace("&amp;", "&")
    return _SPACE.sub(" ", text).strip()


def load_humaid(split: str) -> pd.DataFrame:
    """split: train | validation | test (all 19 disasters combined)."""
    df = pd.read_parquet(DATA / "humaid" / f"{split}.parquet")
    df["text"] = df["tweet_text"].map(clean)
    return df


def load_kerala(split: str = "test") -> pd.DataFrame:
    """Kerala floods 2018 tweets, sorted by tweet id (which follows posting time)."""
    df = pd.read_json(DATA / "kerala" / f"{split}.json", dtype={"tweet_id": str})
    df["text"] = df["tweet_text"].map(clean)
    df["tweet_id"] = df["tweet_id"].astype(int)
    return df.sort_values("tweet_id").reset_index(drop=True)
