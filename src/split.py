import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/golden_eval.csv")
train, test = train_test_split(
    df, test_size=0.35, random_state=42, stratify=df["intent"])
train.to_csv("data/golden_train.csv", index=False)
test.to_csv("data/golden_test.csv", index=False)
print(f"train={len(train)} test={len(test)}")
