"""Segment every DICOM of the cohort with the locked network, measure it and write the image-level
biomarker table used by the classification scripts (Sections 2.5-2.6; one row per image).

usage: python -m measurement.build_feature_table --weights best_unet.pth --index cohort_index.csv --out features.csv

cohort_index.csv must have the columns: path (DICOM file), stem (unique image id), pid (person id),
eye (R/L), label (1 keratoconus, 0 healthy). The person-level split into the development and
independent sets is added by classification/partition.py.
"""
import os, json, argparse
import numpy as np, pandas as pd
import pydicom, torch
from segmentation.model import build_model
from segmentation.infer import segment_dicom
from .features import extract_features, from_original, apply_scale_factor, export_format


def main(weights, index_csv, out_csv, factors_json):
    factors = json.load(open(factors_json))["factors"]
    if any(v is None for v in factors.values()):
        raise SystemExit("config/scale_factors.json: the format-specific scale factors are not distributed with this "
                         "repository; enter the values (available from the authors on request) before building the table.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, weights).eval()
    idx = pd.read_csv(index_csv)
    rows = []
    for _, r in idx.iterrows():
        ds = pydicom.dcmread(r["path"], force=True, stop_before_pixels=True)
        fmt = export_format(int(ds.Columns))
        mask, px, py = segment_dicom(model, r["path"], device)
        f = extract_features(from_original(mask), px, py)
        if f is None:                      # failed the column-level quality filter
            continue
        f = apply_scale_factor(f, fmt, factors)
        f.update({"stem": r["stem"], "pid": r["pid"], "goz": r["eye"], "goz_id": f"{r['pid']}_{r['eye']}",
                  "label": int(r["label"]), "format": fmt})
        rows.append(f)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(len(rows), "images measured ->", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--index", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--factors", default=os.path.join(os.path.dirname(__file__), "..", "config", "scale_factors.json"))
    a = ap.parse_args()
    main(a.weights, a.index, a.out, a.factors)
