"""Download HumAID (all 19 disasters) and the Kerala floods 2018 subset from Hugging Face."""
import urllib.request

from core.data import DATA

HF = "https://huggingface.co/datasets/QCRI"
FILES = {
    **{f"humaid/{s}.parquet": f"{HF}/HumAID-all/resolve/main/data/{s}-00000-of-00001.parquet"
       for s in ("train", "validation", "test")},
    **{f"kerala/{s}.json": f"{HF}/HumAID-events/resolve/main/kerala_floods_2018/{s}.json"
       for s in ("train", "dev", "test")},
}


def main():
    for rel, url in FILES.items():
        path = DATA / rel
        if path.exists():
            print("exists:", rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
        print(f"downloaded: {rel} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
