"""Classification by export format (Sections 2.9 and 3.4; Tables S9 and S10).

Uses the eye-level table and the out-of-fold and independent-set probabilities of the full model
(A+B+C) written by classification/chain.py. An eye is assigned to the export format of the majority
of its sections.

(1) Full model at the carried threshold by export format: eyes, AUC and confusion matrix.
(2) Missing values: eyes with at least one missing biomarker value, by export format and class.
(3) Export format and missing values used alone as predictors of keratoconus: AUC of the
    narrow-format indicator, of the missing-value indicator and of the two combined.
(4) The seven arms and the eight standard parameters retrained and evaluated in wide-format eyes
    alone (LOPO, Youden threshold, final model, wide-format eyes of the independent set), with the
    paired comparisons of the main analysis.
(5) Healthy eyes of both sets: AUC of each biomarker for separating narrow- from wide-format eyes,
    regardless of direction, as a check of the refraction correction (Table S10).

usage: python -m classification.format_analyses --features features_split.csv --results results/ \
           --out results/format_analyses.json
"""
import os, json, argparse
import numpy as np, pandas as pd
from .chain import (load_eye_tables, arm_columns, lopo, final_model, design, auc, auc_ci, paired_auc_difference,
                    youden, confusion, ARMS)
from .compare_classifiers import STD8

GENERATOR = "mt19937"
# generator of the interval of the by-format AUC, as run in the study (see chain.clustered_bootstrap)
BY_FORMAT_GENERATOR = {"dev": "mt19937", "ind": "pcg64"}
FORMATS = ("wide", "narrow")
WIDE_COMPARISONS = [("A+B", "A"), ("A+C", "A"), ("A+B+C", "A"), ("A+B+C", "A+C"), ("A+B+C", "STD8")]


def full_model_outputs(results_dir, g, gb):
    """Threshold and probabilities of the full model saved by classification/chain.py."""
    thr = json.load(open(os.path.join(results_dir, "classification_results.json")))["arms"]["A+B+C"]["threshold"]
    # exact parsing: an eye whose probability equals the threshold must stay on the same side of it
    dev = pd.read_csv(os.path.join(results_dir, "dev_oof_probabilities.csv"), float_precision="round_trip")
    ind = pd.read_csv(os.path.join(results_dir, "independent_probabilities.csv"), float_precision="round_trip")
    assert (dev["goz_id"].astype(str).values == g["goz_id"].astype(str).values).all()
    assert (ind["goz_id"].astype(str).values == gb["goz_id"].astype(str).values).all()
    return thr, dev["p_A+B+C"].to_numpy(), ind["p_A+B+C"].to_numpy()


def by_format(t, p, thr, generator, boot):
    out = {}
    for fmt in FORMATS:
        m = (t["format"] == fmt).values
        sub = t[m].reset_index(drop=True)
        r = {"eyes": int(m.sum()), "kc": int((sub["label"] == 1).sum()), "healthy": int((sub["label"] == 0).sum())}
        if sub["label"].nunique() == 2:
            r["auc"], r["ci"] = auc_ci(sub, p[m], boot, generator)
        r.update(confusion(sub["label"].values, p[m], thr))
        out[fmt] = r
    return out


def missing_matrix(t, cols):
    """[eyes with at least one missing biomarker value, all eyes] by export format and class."""
    any_missing = t[cols].isna().any(axis=1)
    return {"%s_%s" % (fmt, "kc" if y else "healthy"):
            [int(any_missing[(t["format"] == fmt) & (t["label"] == y)].sum()), int(((t["format"] == fmt) & (t["label"] == y)).sum())]
            for fmt in FORMATS for y in (0, 1)}


def indicators(t, cols, boot):
    """AUC of the narrow-format indicator, of the missing-value indicator and of their combination."""
    narrow = (t["format"] == "narrow").to_numpy(float)
    missing = t[cols].isna().any(axis=1).to_numpy(float)
    out = {}
    for name, v in (("narrow_format", narrow), ("any_missing_value", missing), ("combined", narrow + 0.5 * missing)):
        if np.std(v) == 0:
            out[name] = None                      # the indicator is constant in this set
            continue
        a, ci = auc_ci(t, v, boot, GENERATOR)
        out[name] = {"auc": a, "ci": ci}
    return out


def wide_only(g, gb, cols, boot):
    gw = g[g["format"] == "wide"].reset_index(drop=True); bw = gb[gb["format"] == "wide"].reset_index(drop=True)
    y, yb = gw["label"].values, bw["label"].values
    out = {"n": {"dev_eyes": len(gw), "dev_persons": int(gw["pid"].nunique()), "dev_kc": int(y.sum()),
                 "ind_eyes": len(bw), "ind_persons": int(bw["pid"].nunique()), "ind_kc": int(yb.sum())},
           "arms": {}, "comparisons": {}}
    oof, ind = {}, {}
    for name in ARMS + ["STD8"]:
        kol = STD8 if name == "STD8" else arm_columns(cols, name)
        oof[name] = lopo(gw, kol)
        ind[name] = final_model(gw, kol).predict_proba(design(bw, kol))[:, 1]
        thr = youden(y, oof[name])
        a_dev, ci_dev = auc_ci(gw, oof[name], boot, GENERATOR); a_ind, ci_ind = auc_ci(bw, ind[name], boot, GENERATOR)
        out["arms"][name] = {"n_features": len(kol), "threshold": thr,
                             "dev_auc": a_dev, "dev_ci": ci_dev, "dev_youden": confusion(y, oof[name], thr),
                             "ind_auc": a_ind, "ind_ci": ci_ind, "ind_carried": confusion(yb, ind[name], thr)}
        print("wide only %-6s development %.3f [%.3f-%.3f]  independent %.3f [%.3f-%.3f]"
              % (name, a_dev, *ci_dev, a_ind, *ci_ind), flush=True)
    for a1, a0 in WIDE_COMPARISONS:
        out["comparisons"]["%s - %s" % (a1, a0)] = {"dev": paired_auc_difference(gw, oof[a1], oof[a0], boot, GENERATOR),
                                                    "ind": paired_auc_difference(bw, ind[a1], ind[a0], boot, GENERATOR)}
    return out


def format_separation_in_healthy_eyes(g, gb, cols):
    """AUC of each biomarker for narrow- versus wide-format healthy eyes, max(AUC, 1 - AUC)."""
    s = pd.concat([g, gb], ignore_index=True); s = s[s["label"] == 0]
    y = (s["format"] == "narrow").to_numpy(int)
    out = {}
    for c in cols:
        v = s[c].to_numpy(float); m = ~np.isnan(v)
        a = auc(y[m], v[m]); out[c] = float(max(a, 1 - a))
    return {"healthy_eyes": len(s), "narrow": int(y.sum()), "wide": int((1 - y).sum()),
            "auc": dict(sorted(out.items(), key=lambda kv: -kv[1]))}


def run(features_csv, results_dir, out_path, boot=2000):
    g, gb, cols = load_eye_tables(features_csv)
    thr, p_dev, p_ind = full_model_outputs(results_dir, g, gb)
    res = {"threshold": thr,
           "by_format": {"dev": by_format(g, p_dev, thr, BY_FORMAT_GENERATOR["dev"], boot),
                         "ind": by_format(gb, p_ind, thr, BY_FORMAT_GENERATOR["ind"], boot)},
           "missing_values": {"dev": missing_matrix(g, cols), "ind": missing_matrix(gb, cols)},
           "indicators": {"dev": indicators(g, cols, boot), "ind": indicators(gb, cols, boot)},
           "wide_only": wide_only(g, gb, cols, boot),
           "healthy_eyes_format_auc": format_separation_in_healthy_eyes(g, gb, cols)}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("by_format", "missing_values", "indicators")}, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run(a.features, a.results, a.out, a.boot)
