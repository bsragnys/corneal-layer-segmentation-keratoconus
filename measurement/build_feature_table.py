"""Segment every DICOM of the cohort with the locked network, measure it and write the image-level
biomarker table used by the classification scripts (Sections 2.3, 2.5 and 2.6; one row per image).

For each section the export type is determined from the image itself. Refraction-corrected
sections are segmented and measured directly. Uncorrected sections are resampled to the lateral
reference grid, segmented once to locate the anterior surface, corrected by ray tracing, segmented
again with the same locked model and measured from the second mask (measurement/refraction.py).
The thickness biomarkers of both export types are then multiplied by the single common scale
factor (config/scale_factors.json).

usage: python -m measurement.build_feature_table --weights best_unet.pth --index cohort_index.csv --out features.csv

cohort_index.csv must have the columns: path (DICOM file), stem (unique image id), pid (person id),
eye (R/L), label (1 keratoconus, 0 healthy). The person-level split into the development and
independent sets is added by classification/partition.py.
"""
import os, json, argparse
import pandas as pd
import pydicom, torch
from segmentation.model import build_model
from segmentation.infer import segment_dicom, predict_mask
from segmentation.preprocessing import read_dicom, read_dicom_pixels, is_refraction_corrected, PX_REF
from .features import extract_features, from_original, apply_scale_factor, export_format
from .refraction import resample_lateral, crop_window, anterior_surface, correct_refraction


def measure_corrected(model, path, device):
    """Refraction-corrected export: standard preprocessing, one segmentation pass."""
    mask, px, py = segment_dicom(model, path, device)
    return extract_features(from_original(mask), px, py)


def measure_uncorrected(model, path, device):
    """Uncorrected export: first pass for the anterior surface, ray tracing, second pass."""
    img, px, py = read_dicom(path, resample_axial=True)
    img = resample_lateral(img, px)
    window, r0, c0 = crop_window(img)
    first = predict_mask(model, window, device)
    surface = anterior_surface(first, r0, c0, img.shape[1])
    if surface is None:
        return None
    window, _, _ = crop_window(correct_refraction(img, surface, py), r0=r0)
    second = predict_mask(model, window, device)
    return extract_features(second, PX_REF, py)


def main(weights, index_csv, out_csv, factors_json):
    factor = json.load(open(factors_json))["factor"]
    if factor is None:
        raise SystemExit("config/scale_factors.json: the common scale factor is not distributed with this repository; "
                         "enter the value (available from the authors on request) before building the table.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, weights).eval()
    idx = pd.read_csv(index_csv)
    rows = []
    for _, r in idx.iterrows():
        ds = pydicom.dcmread(r["path"], force=True, stop_before_pixels=True)
        corrected = is_refraction_corrected(read_dicom_pixels(r["path"]))
        f = measure_corrected(model, r["path"], device) if corrected else measure_uncorrected(model, r["path"], device)
        if f is None:                      # failed the column-level quality filter
            continue
        f = apply_scale_factor(f, factor)
        f.update({"stem": r["stem"], "pid": r["pid"], "goz": r["eye"], "goz_id": f"{r['pid']}_{r['eye']}",
                  "label": int(r["label"]), "format": export_format(int(ds.Columns)),
                  "refraction_corrected": bool(corrected)})
        rows.append(f)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(len(rows), "images measured ->", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--index", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--factors", default=os.path.join(os.path.dirname(__file__), "..", "config", "scale_factors.json"))
    a = ap.parse_args()
    main(a.weights, a.index, a.out, a.factors)
