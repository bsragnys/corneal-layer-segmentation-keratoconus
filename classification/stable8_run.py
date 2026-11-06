"""Fixed eight-feature STABLE-8 model evaluated in the independent set (Section 2.8; Table 4).

This model was trained in a separate run with the same chain but 300 trees, learning rate 0.05,
median imputation of missing values from the development set and 1000 bootstrap resamples; the
threshold is the Youden point of its own out-of-fold development probabilities.

usage: python -m classification.stable8_run --features features_split.csv
"""
import argparse, json
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from .chain import eye_table, auc, clustered_bootstrap, youden, sens_spec

STABLE8 = ["A_K_oran", "A_pachy_min", "B_epi_z4", "A_tp_dis_merkez_mm", "B_bow_z1", "B_desc_z5", "C_Kant_epi_z2", "C_str_apex_pachy"]
SEED = 42


def model():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                         reg_lambda=1.0, min_child_weight=1, eval_metric="logloss", random_state=SEED, n_jobs=2, verbosity=0)


def impute(X, med):
    X = X.copy(); i = np.where(np.isnan(X)); X[i] = np.take(med, i[1]); return X


def run(features_csv, boot=1000):
    t = pd.read_csv(features_csv)
    g = eye_table(t[t["kume"] == "gelistirme"]); gb = eye_table(t[t["kume"] == "bagimsiz"])
    X = g[STABLE8].to_numpy(float); y = g["label"].to_numpy(float); pid = g["pid"].astype(str).to_numpy()
    med = np.nanmedian(X, axis=0); X = impute(X, med)
    oof = np.full(len(g), np.nan)
    for k in np.unique(pid):
        te = pid == k; tr = ~te
        m = model(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    thr = youden(y, oof)
    m = model(); m.fit(X, y)
    Xb = impute(gb[STABLE8].to_numpy(float), med); yb = gb["label"].to_numpy(float)
    pb = m.predict_proba(Xb)[:, 1]
    bs = [auc(yb[i], pb[i]) for i in clustered_bootstrap(gb["pid"].values, boot)]
    out = {"dev_lopo_auc": auc(y, oof), "threshold": float(thr), "ind_auc": auc(yb, pb),
           "ind_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "ind_sens_spec_carried": sens_spec(yb, pb, thr)}
    print(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--features", required=True)
    run(ap.parse_args().features)
