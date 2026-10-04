"""Compare a keyword baseline against the zero-shot model.
Usage: python eval.py   (needs labeled.csv with columns: sentence,label)
label must be one of the keys in app.RISKS, or 'none'.
"""
import csv
import re
from app import RISKS, LABELS, clf, TEMPLATE, THRESH

BASELINE = {
    "sells personal data": r"\bsell|\bsale\b",
    "shares data with third parties or advertisers": r"third.?part|advertis|partners",
    "tracks users with cookies or trackers": r"cookie|pixel|track",
    "uses user data to train AI models": r"train|machine learning|\bAI\b",
    "keeps data for a long or unlimited time": r"retain|indefinite|as long as",
    "collects precise location": r"location|gps",
    "collects children's data": r"child|minor",
    "lets users delete their data": r"delet|erasure",
}


def base_pred(s):
    for lab, pat in BASELINE.items():
        if re.search(pat, s, re.I):
            return lab
    return "none"


def model_pred(s):
    out = clf()(s, LABELS, hypothesis_template=TEMPLATE, multi_label=True)
    return out["labels"][0] if out["scores"][0] >= THRESH else "none"


rows = list(csv.DictReader(open("labeled.csv", encoding="utf-8")))
n = len(rows)
b = sum(base_pred(r["sentence"]) == r["label"] for r in rows)
m = sum(model_pred(r["sentence"]) == r["label"] for r in rows)
print(f"Labeled clauses: {n}")
print(f"Keyword baseline accuracy: {b / n:.0%}")
print(f"Zero-shot model accuracy:  {m / n:.0%}")
