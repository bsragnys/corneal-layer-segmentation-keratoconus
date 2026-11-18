"""Sensitivity of the classification to a +-1-pixel shift of the Bowman's layer boundaries
(Sections 2.9 and 3.4; Table S7; Figure S2).

Independent set, saved full model (A+B+C) and its locked threshold (written by
classification/chain.py). At the eye level, the Bowman's layer thickness biomarkers (rings 1-5,
apex and global) are changed by one axial pixel of the 9.243 um measurement grid (multiplied by the
common scale factor, as the biomarkers are), either alone, together with a compensating change in
the stroma (posterior boundary shift) or in the epithelium (anterior boundary shift), or with a
random sign for each eye in 1000 draws. The composite indices are updated by the change that the
shift produces in them, and the model is not retrained.

usage: python -m classification.sensitivity_bowman --features features_split.csv --results results/ \
           --out results/sensitivity_bowman.json
"""
import os, json, argparse
import numpy as np
from xgboost import XGBClassifier
from .chain import load_eye_tables, design, auc, auc_ci, paired_auc_difference, confusion
from .format_analyses import full_model_outputs

GENERATOR = "mt19937"
PX_MM = 0.009243                      # axial sampling interval of the measurement grid (mm)
BOW = ["B_bow_z1", "B_bow_z2", "B_bow_z3", "B_bow_z4", "B_bow_z5", "B_bow_apex", "B_bow_global"]
STR = [c.replace("bow", "str") for c in BOW]
EPI = [c.replace("bow", "epi") for c in BOW]


def composites(t, factor):
    """Composite indices from the eye-level biomarkers (the keratometry x epithelium products use
    the unscaled epithelial thickness)."""
    return {"C_bow_apex_global": t.B_bow_apex / t.B_bow_global,
            "C_str_apex_pachy": t.B_str_apex / t.A_pachy_apex,
            "C_epi_apex_pachy": t.B_epi_apex / t.A_pachy_apex,
            "C_epi_remodel": (t.B_epi_global - t.B_epi_apex) / t.B_epi_apex,
            "C_Kant_epi_apex": t.A_K_ant * t.B_epi_apex / factor,
            "C_Kant_epi_z2": t.A_K_ant * t.B_epi_z2 / factor}


def shift(t, mode, sign, factor):
    """mode: 'bowman' (thickness only), 'posterior' (compensated in the stroma) or 'anterior'
    (compensated in the epithelium); sign: +1 / -1 or one sign per eye."""
    p = t.copy(); delta = np.asarray(sign, float) * PX_MM * factor
    for c in BOW:
        p[c] = p[c] + delta
    if mode == "posterior":
        for c in STR:
            p[c] = p[c] - delta
    if mode == "anterior":
        for c in EPI:
            p[c] = p[c] - delta
    before, after = composites(t, factor), composites(p, factor)
    for c in before:
        d = (after[c] - before[c]).to_numpy(); ok = np.isfinite(d) & p[c].notna().to_numpy()
        p.loc[ok, c] = p.loc[ok, c] + d[ok]
    return p


def run(features_csv, results_dir, out_path, factor, boot=2000, draws=1000):
    g, gb, cols = load_eye_tables(features_csv)
    thr, _, p0 = full_model_outputs(results_dir, g, gb)
    model = XGBClassifier(); model.load_model(os.path.join(results_dir, "model_A+B+C.json"))
    assert np.abs(model.predict_proba(design(gb, cols))[:, 1] - p0).max() < 1e-6
    yb = gb["label"].values
    a0, ci0 = auc_ci(gb, p0, boot, GENERATOR)
    res = {"threshold": thr, "no_shift": {"auc": a0, "ci": ci0, **confusion(yb, p0, thr)}, "scenarios": {}}
    for mode in ("bowman", "posterior", "anterior"):
        for sign in (+1, -1):
            p1 = model.predict_proba(design(shift(gb, mode, sign, factor), cols))[:, 1]
            a1, ci1 = auc_ci(gb, p1, boot, GENERATOR)
            d, d_ci, _ = paired_auc_difference(gb, p1, p0, boot, GENERATOR)
            res["scenarios"]["%s_%+d" % (mode, sign)] = {"auc": a1, "ci": ci1, "delta_auc": d, "delta_ci": d_ci,
                                                         **confusion(yb, p1, thr)}
            print("%-9s %+d  AUC %.3f [%.3f-%.3f]  change %+.3f [%+.3f; %+.3f]" % (mode, sign, a1, *ci1, d, *d_ci), flush=True)
    rng = np.random.default_rng(2026); aucs, sens, spec = [], [], []
    for _ in range(draws):
        p1 = model.predict_proba(design(shift(gb, "bowman", rng.choice([-1.0, 1.0], len(gb)), factor), cols))[:, 1]
        c = confusion(yb, p1, thr)
        aucs.append(auc(yb, p1)); sens.append(c["sens"]); spec.append(c["spec"])
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res["random_sign_per_eye"] = {"auc_median": float(np.median(aucs)), "auc_2.5_97.5": pct(aucs),
                                  "sens_median": float(np.median(sens)), "sens_2.5_97.5": pct(sens),
                                  "spec_median": float(np.median(spec)), "spec_2.5_97.5": pct(spec)}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res["random_sign_per_eye"], indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--factors", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "scale_factors.json"))
    a = ap.parse_args()
    factor = json.load(open(a.factors))["factor"]
    if factor is None:
        raise SystemExit("config/scale_factors.json: the common scale factor is not distributed with this repository; "
                         "enter the value (available from the authors on request) before running this script.")
    run(a.features, a.results, a.out, factor, a.boot)
