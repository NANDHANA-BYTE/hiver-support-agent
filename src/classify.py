"""
Three classifiers, compared head-to-head in eval/run_eval.py:

  1. TrivialBaseline   - always predicts the majority class from train
  2. RuleBaseline       - the keyword/regex labeler from weak_label.py, used
                          here AS A CLASSIFIER (not as ground truth)
  3. TfidfLogReg        - our system: TF-IDF + multinomial Logistic
                          Regression, trained on golden_train.csv

An LLM-based classifier (llm_client.py) can be swapped in if an API key is
present; see decision_log.md #6 for why it isn't the default here.
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from weak_label import classify_intent as _rule_classify_intent


class TrivialBaseline:
    def fit(self, X, y):
        self.majority = y.value_counts().idxmax()
        return self

    def predict(self, X):
        return [self.majority] * len(X)


class RuleBaseline:
    """Wraps the weak_label.py regex rules as a classifier for comparison."""
    def fit(self, X, y):
        return self  # stateless, no training

    def predict(self, X):
        return [_rule_classify_intent(t) for t in X]


class TfidfLogReg:
    def __init__(self):
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_df=0.9,
                sublinear_tf=True, stop_words="english")),
            ("clf", LogisticRegression(
                max_iter=2000, C=3.0, class_weight="balanced")),
        ])

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict(X)

    def predict_proba_top(self, X):
        """Returns (pred, confidence) pairs."""
        probs = self.pipe.predict_proba(X)
        classes = self.pipe.classes_
        preds, confs = [], []
        for row in probs:
            i = row.argmax()
            preds.append(classes[i])
            confs.append(float(row[i]))
        return preds, confs

    def save(self, path="data/model.joblib"):
        joblib.dump(self.pipe, path)

    @classmethod
    def load(cls, path="data/model.joblib"):
        obj = cls()
        obj.pipe = joblib.load(path)
        return obj


def train_and_save():
    train = pd.read_csv("data/golden_train.csv")
    model = TfidfLogReg().fit(train["customer_clean"], train["intent"])
    model.save()
    print("model trained on", len(train), "rows and saved to data/model.joblib")
    return model


if __name__ == "__main__":
    train_and_save()
