"""
Tags each row of the raw sample with a detected language. Used to scope the
golden set to English only (decision_log.md #1).
"""
import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic detection


def safe_detect(text):
    try:
        return detect(text)
    except Exception:
        return "unk"


def main():
    df = pd.read_csv("data/raw_sample.csv")
    df["lang"] = df["customer_clean"].apply(safe_detect)
    df.to_csv("data/with_lang.csv", index=False)
    print(df["lang"].value_counts())


if __name__ == "__main__":
    main()
