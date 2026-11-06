"""Write the eye-level biomarker table of the development set for the R script (univariate_tests.R).
usage: python -m classification.export_eye_table --features features_split.csv --out eye_table_development.csv
"""
import argparse
import pandas as pd
from .chain import eye_table

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = pd.read_csv(a.features)
    eye_table(t[t["kume"] == "gelistirme"]).to_csv(a.out, index=False)
