"""Invariance of the classification to the common scale factor (Section 2.7).

The thickness biomarkers (the four pachymetric biomarkers of group A and all biomarkers of group B)
are multiplied by a constant, the leave-one-person-out predictions and the final model of the full
arm (A+B+C) are computed again, and the largest absolute difference between the two sets of
probabilities is reported. With --remove-scale-factor the constant is the reciprocal of the common
scale factor in config/scale_factors.json, that is, the raw mask thicknesses are used, as in the
paper; any other constant can be given with --multiplier.

usage: python -m classification.scale_invariance --features features_split.csv --remove-scale-factor
       python -m classification.scale_invariance --features features_split.csv --multiplier 0.9
"""
import os, json, argparse
import numpy as np
from .chain import load_eye_tables, lopo, final_model, design

SCALED_A = ("A_pachy_min", "A_pachy_apex", "A_pachy_global", "A_pachy_asimetri")


def thickness_columns(cols):
    return [c for c in cols if c.startswith("B_") or c in SCALED_A]


def run(features_csv, multiplier):
    g, gb, cols = load_eye_tables(features_csv)
    mm = thickness_columns(cols)
    oof = lopo(g, cols); ind = final_model(g, cols).predict_proba(design(gb, cols))[:, 1]
    g2, gb2 = g.copy(), gb.copy()
    g2[mm] = g2[mm] * multiplier; gb2[mm] = gb2[mm] * multiplier
    oof2 = lopo(g2, cols); ind2 = final_model(g2, cols).predict_proba(design(gb2, cols))[:, 1]
    res = {"thickness_biomarkers": len(mm), "eyes": len(g) + len(gb),
           "max_abs_difference_dev_oof": float(np.abs(oof2 - oof).max()),
           "max_abs_difference_ind": float(np.abs(ind2 - ind).max())}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--multiplier", type=float)
    ap.add_argument("--remove-scale-factor", action="store_true")
    ap.add_argument("--factors", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "scale_factors.json"))
    a = ap.parse_args()
    if a.remove_scale_factor:
        factor = json.load(open(a.factors))["factor"]
        if factor is None:
            raise SystemExit("config/scale_factors.json: the common scale factor is not distributed with this repository; "
                             "enter the value (available from the authors on request) or use --multiplier.")
        a.multiplier = 1.0 / factor
    if a.multiplier is None:
        raise SystemExit("give --multiplier or --remove-scale-factor")
    run(a.features, a.multiplier)
