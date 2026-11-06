"""Comparison with standard parameters only and with other classifiers (Section 2.9; Section 3.4; Table S8).

Same eye-level table, same person-based leave-one-person-out design, same Youden thresholds carried
unchanged to the independent set and the same person-clustered bootstrap as the main chain. Models:

  A+B+C      gradient-boosted trees of the main chain (200 trees, depth 3, learning rate 0.1,
             missing values coded as -999), all 49 biomarkers
  A+C        the same classifier without the layer x ring thicknesses (17 biomarkers)
  STD8       the same classifier with the eight standard pachymetry and keratometry parameters only
             (thinnest, apex and global pachymetry, anterior and posterior keratometry and their ratio,
             thinnest-point distance from the image centre, pachymetric asymmetry)
  STABLE-8   the fixed eight-feature core with the main-chain settings (200 trees, learning rate 0.1)
             and with the settings of the separate run (300 trees, learning rate 0.05)
  LR         L2-regularised logistic regression (C = 1), median imputation, standardised inputs, 49 biomarkers
  RF         random forest (300 trees, minimum leaf size 2), median imputation, 49 biomarkers

Differences in AUC are reported relative to A+B+C with person-clustered bootstrap intervals.

usage: python -m classification.compare_classifiers --features features_split.csv --out results/table_s8.json
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .chain import eye_table, feature_columns, auc, auc_ci, paired_auc_difference, youden, sens_spec, MISSING, SEED

STD8 = ["A_pachy_min", "A_pachy_apex", "A_pachy_global", "A_K_ant", "A_K_post", "A_K_oran",
        "A_tp_dis_merkez_mm", "A_pachy_asimetri"]
STABLE8 = ["A_K_oran", "A_pachy_min", "B_epi_z4", "A_tp_dis_merkez_mm", "B_bow_z1", "B_desc_z5",
           "C_Kant_epi_z2", "C_str_apex_pachy"]


def xgb(n_estimators=200, learning_rate=0.1):
    return XGBClassifier(n_estimators=n_estimators, max_depth=3, learning_rate=learning_rate, subsample=0.9,
                         colsample_bytree=0.9, random_state=SEED, n_jobs=2, eval_metric="logloss", verbosity=0)


def models(cols):
    a_c = [c for c in cols if c[0] in "AC"]
    return {
        "A+B+C": (cols, lambda: xgb(), True),
        "A+C": (a_c, lambda: xgb(), True),
        "STD8": (STD8, lambda: xgb(), True),
        "STABLE-8 (200, 0.1)": (STABLE8, lambda: xgb(), True),
        "STABLE-8 (300, 0.05)": (STABLE8, lambda: xgb(300, 0.05), True),
        "LR": (cols, lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                           LogisticRegression(C=1.0, max_iter=5000)), False),
        "RF": (cols, lambda: make_pipeline(SimpleImputer(strategy="median"),
                                           RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                                  random_state=SEED, n_jobs=2)), False),
    }


def _x(frame, cols, code_missing):
    x = frame[cols].to_numpy(float)
    return np.nan_to_num(x, nan=MISSING) if code_missing else x


def lopo_probabilities(g, cols, make, code_missing):
    p = np.full(len(g), np.nan); y = g["label"].to_numpy(float); pid = g["pid"].astype(str).to_numpy()
    for person in np.unique(pid):
        te = pid == person
        m = make().fit(_x(g[~te], cols, code_missing), y[~te])
        p[te] = m.predict_proba(_x(g[te], cols, code_missing))[:, 1]
    return p


def run(features_csv, out_path, boot=2000):
    t = pd.read_csv(features_csv)
    g = eye_table(t[t["kume"] == "gelistirme"]); gb = eye_table(t[t["kume"] == "bagimsiz"])
    cols = feature_columns(t)
    y, yb = g["label"].to_numpy(float), gb["label"].to_numpy(float)
    res, oof, ind = {}, {}, {}
    for name, (kol, make, code_missing) in models(cols).items():
        oof[name] = lopo_probabilities(g, kol, make, code_missing)
        ind[name] = make().fit(_x(g, kol, code_missing), y).predict_proba(_x(gb, kol, code_missing))[:, 1]
        thr = youden(y, oof[name])
        a_dev, ci_dev = auc_ci(g, oof[name], boot); a_ind, ci_ind = auc_ci(gb, ind[name], boot)
        res[name] = {"n_features": len(kol), "dev_lopo_auc": a_dev, "dev_ci": list(ci_dev),
                     "dev_sens_spec": sens_spec(y, oof[name], thr), "threshold": float(thr),
                     "ind_auc": a_ind, "ind_ci": list(ci_ind), "ind_sens_spec_carried": sens_spec(yb, ind[name], thr)}
        print(name, "development LOPO AUC %.3f [%.3f-%.3f]  independent AUC %.3f [%.3f-%.3f]"
              % (a_dev, *ci_dev, a_ind, *ci_ind))
    for name in res:
        if name == "A+B+C":
            continue
        d_dev, ci_d = paired_auc_difference(g, oof["A+B+C"], oof[name], boot)
        d_ind, ci_i = paired_auc_difference(gb, ind["A+B+C"], ind[name], boot)
        res[name]["delta_vs_ABC_dev"] = [d_dev, list(ci_d)]; res[name]["delta_vs_ABC_ind"] = [d_ind, list(ci_i)]
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run(a.features, a.out, a.boot)
