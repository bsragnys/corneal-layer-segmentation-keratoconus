"""Keratoconus classification and validation design (Section 2.8; Table 4; Figure 5; Figure 6B).

Eye-level table: image-level biomarkers averaged per eye; an eye is assigned to the export format
of the majority of its sections. Classifier: XGBoost (200 trees, maximum depth 3, learning rate
0.1, row and column subsampling 0.9, missing values coded as -999, seed 42). The hyperparameters
are fixed and not tuned. Seven feature arms are formed from the three biomarker groups: A (11),
B (32), C (6), A+B (43), A+C (17), B+C (38) and A+B+C (49); there is no data-driven feature
selection.

Development set: person-based leave-one-person-out (LOPO) cross-validation; all eyes of one person
are held out in each fold. For each arm the threshold is the Youden point of the pooled
out-of-fold probabilities, and the final model is trained on the whole development set.
Independent set: a single evaluation with the final models and the thresholds carried unchanged.

Confidence intervals: person-clustered bootstrap (2000 resamples). Added value of the new
biomarkers: paired person-clustered bootstrap differences in AUC for the three comparisons defined
in advance (A+B - A, A+C - A, A+B+C - A) and for A+B+C - A+C. SHAP: tree algorithm, computed in
each LOPO fold only for the held-out person, so that the explanations are out-of-fold.

usage: python -m classification.chain --features features_split.csv --out-dir results/ [--shap]

Written to the output directory: classification_results.json, dev_oof_probabilities.csv,
independent_probabilities.csv and the final model of every arm (model_<arm>.json); with --shap
also shap_importance.csv and shap_values_oof.npy.
"""
import os, json, argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 42
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                  random_state=SEED, eval_metric="logloss")
MISSING = -999
ARMS = ["A", "B", "C", "A+B", "A+C", "B+C", "A+B+C"]
COMPARISONS = [("A+B", "A"), ("A+C", "A"), ("A+B+C", "A"), ("A+B+C", "A+C")]


def feature_columns(df):
    return [c for c in df.columns if c[:2] in ("A_", "B_", "C_")]


def arm_columns(cols, arm):
    """Biomarkers of an arm, in the column order of the table."""
    groups = arm.split("+")
    return [c for c in cols if c[0] in groups]


def eye_table(df):
    agg = {**{c: "mean" for c in feature_columns(df)}, "label": "first", "pid": "first"}
    if "format" in df.columns:
        agg["format"] = lambda s: s.mode()[0]
    return df.groupby("goz_id").agg(agg).reset_index()


def load_eye_tables(features_csv):
    """Eye-level tables of the development and the independent set and the biomarker columns."""
    t = pd.read_csv(features_csv)
    return eye_table(t[t["kume"] == "gelistirme"]), eye_table(t[t["kume"] == "bagimsiz"]), feature_columns(t)


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(**XGB_PARAMS)


def design(g, cols):
    return np.nan_to_num(g[cols].values.astype(float), nan=MISSING)


def lopo(g, cols, make=_xgb, x=design):
    """Out-of-fold probabilities of the development set (one fold per person)."""
    p = np.full(len(g), np.nan)
    for person in g["pid"].unique():
        te = (g["pid"] == person).values
        mdl = make()
        mdl.fit(x(g[~te], cols), g["label"].values[~te])
        p[te] = mdl.predict_proba(x(g[te], cols))[:, 1]
    return p


def final_model(g, cols, make=_xgb, x=design):
    mdl = make()
    mdl.fit(x(g, cols), g["label"].values)
    return mdl


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def clustered_bootstrap(pid, boot=2000, seed=SEED, generator="pcg64"):
    """Person-clustered bootstrap: persons are drawn with replacement and all eyes of a drawn
    person are taken. Yields one index array per resample.

    generator="pcg64": numpy Generator, persons in sorted order (the seven arms and their paired
    comparisons). generator="mt19937": numpy RandomState, persons in order of appearance (the
    analyses in compare_classifiers, format_analyses, subset_analyses and sensitivity_bowman).
    The two were used in different runs of the study; both are kept so that every reported
    interval can be reproduced. They differ only in the random numbers drawn."""
    pid = np.asarray(pid)
    if generator == "pcg64":
        rng = np.random.default_rng(seed); persons = np.unique(pid)
    else:
        rng = np.random.RandomState(seed); persons = pd.unique(pid)
    index = {k: np.where(pid == k)[0] for k in persons}
    for _ in range(boot):
        yield np.concatenate([index[k] for k in rng.choice(persons, len(persons), replace=True)])


def _interval(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def auc_ci(g, p, boot=2000, generator="pcg64"):
    y = g["label"].values; p = np.asarray(p)
    bs = [auc(y[i], p[i]) for i in clustered_bootstrap(g["pid"].values, boot, generator=generator)]
    return float(auc(y, p)), _interval([b for b in bs if np.isfinite(b)])


def paired_auc_difference(g, p1, p0, boot=2000, generator="pcg64"):
    """AUC(p1) - AUC(p0) with its paired person-clustered bootstrap interval and two-sided P value."""
    y = g["label"].values; p1 = np.asarray(p1); p0 = np.asarray(p0); d = []
    for i in clustered_bootstrap(g["pid"].values, boot, generator=generator):
        a1, a0 = auc(y[i], p1[i]), auc(y[i], p0[i])
        if np.isfinite(a1) and np.isfinite(a0):
            d.append(a1 - a0)
    d = np.array(d)
    p_value = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(auc(y, p1) - auc(y, p0)), _interval(d), float(p_value)


def confusion(y, p, thr):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((p >= thr) & (y == 1)).sum()); fn = int(((p < thr) & (y == 1)).sum())
    tn = int(((p < thr) & (y == 0)).sum()); fp = int(((p >= thr) & (y == 0)).sum())
    return {"sens": tp / (tp + fn) if tp + fn else None, "spec": tn / (tn + fp) if tn + fp else None,
            "TP": tp, "FN": fn, "TN": tn, "FP": fp}


def sens_spec(y, p, thr):
    c = confusion(y, p, thr)
    return c["sens"], c["spec"]


def rate_ci(g, p, thr, target, boot=2000, generator="mt19937"):
    """Sensitivity (target=1) or specificity (target=0) at a threshold with its person-clustered
    bootstrap interval."""
    y = g["label"].values; hit = ((np.asarray(p) >= thr).astype(int) == target)
    bs = []
    for i in clustered_bootstrap(g["pid"].values, boot, generator=generator):
        m = y[i] == target
        if m.sum():
            bs.append(hit[i][m].mean())
    return float(hit[y == target].mean()), _interval(bs)


def youden(y, p):
    """Threshold with the largest sensitivity + specificity - 1 (the lowest one if several tie)."""
    best, thr = -1, 0.5
    for e in np.unique(p):
        s, sp = sens_spec(y, p, e)
        if s is None or sp is None:
            continue
        if s + sp - 1 > best + 1e-12:
            best, thr = s + sp - 1, e
    return float(thr)


def run(features_csv, out_dir, boot=2000):
    os.makedirs(out_dir, exist_ok=True)
    g, gb, cols = load_eye_tables(features_csv)
    y, yb = g["label"].values, gb["label"].values
    res = {"n_dev": {"eyes": len(g), "persons": int(g["pid"].nunique())},
           "n_ind": {"eyes": len(gb), "persons": int(gb["pid"].nunique())}, "arms": {}, "comparisons": {}}
    oof, ind = {}, {}
    for arm in ARMS:
        kol = arm_columns(cols, arm)
        oof[arm] = lopo(g, kol)
        thr = youden(y, oof[arm])
        mdl = final_model(g, kol)
        mdl.save_model(os.path.join(out_dir, "model_%s.json" % arm))
        ind[arm] = mdl.predict_proba(design(gb, kol))[:, 1]
        a_dev, ci_dev = auc_ci(g, oof[arm], boot); a_ind, ci_ind = auc_ci(gb, ind[arm], boot)
        res["arms"][arm] = {"n_features": len(kol), "features": kol, "threshold": thr,
                            "dev_auc": a_dev, "dev_ci": ci_dev, "dev_youden": confusion(y, oof[arm], thr),
                            "ind_auc": a_ind, "ind_ci": ci_ind, "ind_carried": confusion(yb, ind[arm], thr)}
        print("%-6s n=%2d  development %.3f [%.3f-%.3f]  independent %.3f [%.3f-%.3f]  threshold %.3f"
              % (arm, len(kol), a_dev, *ci_dev, a_ind, *ci_ind, thr), flush=True)
    for a1, a0 in COMPARISONS:
        res["comparisons"]["%s - %s" % (a1, a0)] = {"dev": paired_auc_difference(g, oof[a1], oof[a0], boot),
                                                    "ind": paired_auc_difference(gb, ind[a1], ind[a0], boot)}
    with open(os.path.join(out_dir, "classification_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    for name, t, p in (("dev_oof_probabilities.csv", g, oof), ("independent_probabilities.csv", gb, ind)):
        out = pd.DataFrame({"goz_id": t["goz_id"], "pid": t["pid"], "label": t["label"]})
        if "format" in t.columns:
            out["format"] = t["format"]
        for arm in ARMS:
            out["p_" + arm] = p[arm]
        out.to_csv(os.path.join(out_dir, name), index=False)
    for k, v in res["comparisons"].items():
        print("%-14s development %+.3f [%+.3f; %+.3f] P %.3f | independent %+.3f [%+.3f; %+.3f] P %.3f"
              % (k, v["dev"][0], *v["dev"][1], v["dev"][2], v["ind"][0], *v["ind"][1], v["ind"][2]))
    return res


def shap_importance(features_csv, out_dir):
    """Out-of-fold SHAP values of the A+B+C model (Section 2.9; Figure 6B, Figure S1)."""
    import shap
    g, _, cols = load_eye_tables(features_csv)
    S = np.zeros((len(g), len(cols)))
    for person in g["pid"].unique():
        te = (g["pid"] == person).values
        mdl = final_model(g[~te], cols)
        S[te] = shap.TreeExplainer(mdl).shap_values(design(g[te], cols))
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(S).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(out_dir, "shap_importance.csv"), index=False)
    np.save(os.path.join(out_dir, "shap_values_oof.npy"), S)
    print(imp.head(12).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--shap", action="store_true", help="also compute the out-of-fold SHAP importance")
    a = ap.parse_args()
    run(a.features, a.out_dir, a.boot)
    if a.shap:
        shap_importance(a.features, a.out_dir)
