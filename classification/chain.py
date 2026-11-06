"""Keratoconus classification and validation design (Section 2.8; Table 4; Figure 5).

Eye-level table: image-level biomarkers averaged per eye. Development set: person-based
leave-one-person-out (LOPO) cross-validation with XGBoost (200 trees, maximum depth 3, learning
rate 0.1, row and column subsampling 0.9, missing values coded as -999, seed 42). Feature sets:
A+C (17), A+B+C (49) and the within-fold selection arm (univariate AUC ranking inside the training
fold, removal of features with |r| > 0.90 to an already selected feature, at most 12 features).
Features selected in every fold form the STABLE-8 core. Thresholds: Youden index on the
out-of-fold probabilities of the development set, carried unchanged to the independent set.
Independent set: a single evaluation of the classifiers trained on the whole development set.
Confidence intervals: person-clustered bootstrap (2000 resamples); paired differences between
feature sets by the same bootstrap. SHAP: TreeExplainer on the out-of-fold predictions.

usage: python -m classification.chain --features features_split.csv --out-dir results/
"""
import os, json, argparse
import numpy as np, pandas as pd
from scipy import stats

SEED = 42
XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                  random_state=SEED, eval_metric="logloss")
MISSING = -999


def feature_columns(df):
    return [c for c in df.columns if c[:2] in ("A_", "B_", "C_")]


def eye_table(df):
    agg = {**{c: "mean" for c in feature_columns(df)}, "label": "first", "pid": "first"}
    if "format" in df.columns:
        agg["format"] = lambda s: s.mode()[0]
    return df.groupby("goz_id").agg(agg).reset_index()


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(**XGB_PARAMS)


def within_fold_selection(tr, cols, k_max=12, r_thr=0.9):
    aucs = {}; y = tr["label"].values
    for c in cols:
        v = tr[c].values; m = np.isfinite(v)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        try:
            u = stats.mannwhitneyu(v[m][y[m] == 1], v[m][y[m] == 0])
            a = u.statistic / ((y[m] == 1).sum() * (y[m] == 0).sum())
            aucs[c] = max(a, 1 - a)
        except Exception:
            continue
    chosen = []
    for c in sorted(aucs, key=aucs.get, reverse=True):
        redundant = False
        for s in chosen:
            v1, v2 = tr[c].values, tr[s].values
            m = np.isfinite(v1) & np.isfinite(v2)
            if m.sum() > 10 and abs(np.corrcoef(v1[m], v2[m])[0, 1]) > r_thr:
                redundant = True; break
        if not redundant:
            chosen.append(c)
        if len(chosen) >= k_max:
            break
    return chosen


def lopo(g, cols, select=False):
    """Out-of-fold probabilities; returns (p, selection counts per feature)."""
    p = np.full(len(g), np.nan); counts = {}
    for person in g["pid"].unique():
        tr = g[g["pid"] != person]; te_idx = g.index[g["pid"] == person]
        kol = within_fold_selection(tr, cols) if select else cols
        for c in kol:
            counts[c] = counts.get(c, 0) + 1
        mdl = _xgb()
        mdl.fit(np.nan_to_num(tr[kol].values, nan=MISSING), tr["label"].values)
        p[te_idx] = mdl.predict_proba(np.nan_to_num(g.loc[te_idx, kol].values, nan=MISSING))[:, 1]
    return p, counts


def auc(y, p):
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (stats.rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def clustered_bootstrap(pid, boot=2000, seed=SEED):
    rng = np.random.default_rng(seed); persons = np.unique(pid)
    for _ in range(boot):
        sel = rng.choice(persons, len(persons), replace=True)
        yield np.concatenate([np.where(pid == k)[0] for k in sel])


def auc_ci(g, p, boot=2000):
    y = g["label"].values; pid = g["pid"].values
    bs = [auc(y[i], p[i]) for i in clustered_bootstrap(pid, boot)]
    bs = [b for b in bs if np.isfinite(b)]
    return auc(y, p), (np.percentile(bs, 2.5), np.percentile(bs, 97.5))


def paired_auc_difference(g, p1, p0, boot=2000):
    y = g["label"].values; pid = g["pid"].values
    d = [auc(y[i], p1[i]) - auc(y[i], p0[i]) for i in clustered_bootstrap(pid, boot)]
    d = [x for x in d if np.isfinite(x)]
    return auc(y, p1) - auc(y, p0), (np.percentile(d, 2.5), np.percentile(d, 97.5))


def sens_spec(y, p, thr):
    tp = ((p >= thr) & (y == 1)).sum(); fn = ((p < thr) & (y == 1)).sum()
    tn = ((p < thr) & (y == 0)).sum(); fp = ((p >= thr) & (y == 0)).sum()
    return tp / (tp + fn) if tp + fn else np.nan, tn / (tn + fp) if tn + fp else np.nan


def youden(y, p):
    best, thr = -1, 0.5
    for e in np.unique(p):
        s, sp = sens_spec(y, p, e)
        if np.isfinite(s) and np.isfinite(sp) and s + sp - 1 > best:
            best, thr = s + sp - 1, e
    return thr


def run(features_csv, out_dir, boot=2000):
    os.makedirs(out_dir, exist_ok=True)
    t = pd.read_csv(features_csv)
    dev = t[t["kume"] == "gelistirme"]; ind = t[t["kume"] == "bagimsiz"]
    g = eye_table(dev); gb = eye_table(ind); cols = feature_columns(dev)
    sets = {"A+C": [c for c in cols if c[0] in "AC"], "A+B+C": cols}
    res = {}; probs = {}
    for name, kol in sets.items():
        p, _ = lopo(g, kol); probs[name] = p
        a, ci = auc_ci(g, p, boot); thr = youden(g["label"].values, p)
        res[name] = {"dev_auc": a, "dev_ci": ci, "threshold": thr, "dev_sens_spec_youden": sens_spec(g["label"].values, p, thr)}
    p_sel, counts = lopo(g, cols, select=True); probs["within-fold selection"] = p_sel
    a, ci = auc_ci(g, p_sel, boot); thr = youden(g["label"].values, p_sel)
    res["within-fold selection"] = {"dev_auc": a, "dev_ci": ci, "threshold": thr, "dev_sens_spec_youden": sens_spec(g["label"].values, p_sel, thr)}
    n_folds = g["pid"].nunique()
    stable = sorted([c for c, n in counts.items() if n == n_folds])
    res["STABLE_core"] = stable
    res["paired_dev"] = {"A+B+C minus A+C": paired_auc_difference(g, probs["A+B+C"], probs["A+C"], boot),
                         "A+B+C minus within-fold selection": paired_auc_difference(g, probs["A+B+C"], p_sel, boot)}
    # independent set: classifiers trained on the whole development set, thresholds carried unchanged
    pind = {}
    for name, kol in sets.items():
        mdl = _xgb(); mdl.fit(np.nan_to_num(g[kol].values, nan=MISSING), g["label"].values)
        p = mdl.predict_proba(np.nan_to_num(gb[kol].values, nan=MISSING))[:, 1]; pind[name] = p
        a, ci = auc_ci(gb, p, boot)
        res[name].update({"ind_auc": a, "ind_ci": ci, "ind_sens_spec_carried": sens_spec(gb["label"].values, p, res[name]["threshold"])})
        if "format" in gb.columns:
            for fmt in sorted(gb["format"].unique()):
                sub = gb["format"] == fmt
                res[name][f"ind_{fmt}"] = {"n": int(sub.sum()), "sens_spec_carried": sens_spec(gb["label"].values[sub], p[sub], res[name]["threshold"])}
                if gb["label"][sub].nunique() == 2:
                    res[name][f"ind_{fmt}"]["auc"], res[name][f"ind_{fmt}"]["ci"] = auc_ci(gb[sub].reset_index(drop=True), p[sub], boot)
    res["paired_ind"] = {"A+B+C minus A+C": paired_auc_difference(gb, pind["A+B+C"], pind["A+C"], boot)}
    json.dump(res, open(os.path.join(out_dir, "classification_results.json"), "w"), indent=1, default=float)
    pd.DataFrame({"goz_id": g["goz_id"], "pid": g["pid"], "label": g["label"], **{f"p_{k}": v for k, v in probs.items()}}).to_csv(os.path.join(out_dir, "dev_oof_probabilities.csv"), index=False)
    pd.DataFrame({"goz_id": gb["goz_id"], "pid": gb["pid"], "label": gb["label"], **{f"p_{k}": v for k, v in pind.items()}}).to_csv(os.path.join(out_dir, "independent_probabilities.csv"), index=False)
    print(json.dumps(res, indent=1, default=float))


def shap_importance(features_csv, out_dir):
    """Out-of-fold SHAP values of the A+B+C model (Section 2.9; Figure 6B)."""
    import shap
    t = pd.read_csv(features_csv); g = eye_table(t[t["kume"] == "gelistirme"]); cols = feature_columns(t)
    S = np.zeros((len(g), len(cols)))
    for person in g["pid"].unique():
        tr = g[g["pid"] != person]; te_idx = g.index[g["pid"] == person]
        mdl = _xgb(); mdl.fit(np.nan_to_num(tr[cols].values, nan=MISSING), tr["label"].values)
        S[te_idx] = shap.TreeExplainer(mdl).shap_values(np.nan_to_num(g.loc[te_idx, cols].values, nan=MISSING))
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(S).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(out_dir, "shap_importance.csv"), index=False)
    np.save(os.path.join(out_dir, "shap_values_oof.npy"), S)
    print(imp.head(12).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shap", action="store_true", help="also compute out-of-fold SHAP importance")
    a = ap.parse_args()
    run(a.features, a.out_dir)
    if a.shap:
        shap_importance(a.features, a.out_dir)
