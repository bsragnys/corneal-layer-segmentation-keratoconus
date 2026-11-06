"""Person-level split of the classification cohort into the development and the independent set
(Section 2.2): a fixed rule computed from the person identifier. The SHA-1 hash of the identifier
string is taken as an integer; an even value assigns the person to the development set and an odd
value to the independent set. All eyes, series and sections of a person follow the person.

usage: python -m classification.partition --features features.csv --out features_split.csv
"""
import hashlib, argparse
import pandas as pd


def development(person_id):
    return int(hashlib.sha1(str(person_id).encode()).hexdigest(), 16) % 2 == 0


def add_split(df):
    persons = df.groupby("pid").agg(label=("label", "first")).reset_index()
    persons["kume"] = ["gelistirme" if development(k) else "bagimsiz" for k in persons["pid"]]
    return df.merge(persons[["pid", "kume"]], on="pid", how="left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = add_split(pd.read_csv(a.features))
    t.to_csv(a.out, index=False)
    print(t.groupby("kume").agg(persons=("pid", "nunique"), eyes=("goz_id", "nunique"), images=("stem", "count")))
