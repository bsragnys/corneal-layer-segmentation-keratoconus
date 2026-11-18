"""Standard parameters alone and other classifiers (Sections 2.9 and 3.4; Table 4, last row; Table S8).

Same eye-level table, same person-based leave-one-person-out design, the Youden threshold of each
model carried unchanged to the independent set and the person-clustered bootstrap. Models:

  A+B+C   gradient-boosted trees of the main analysis, all 49 biomarkers
  A       the same classifier with the standard tomographic biomarkers (11)
  STD8    the same classifier with the eight standard pachymetric and keratometric parameters alone:
          thinnest, apex and global pachymetry, anterior and posterior keratometry and their ratio,
          the distance of the thinnest point from the image centre and pachymetric left-right
          asymmetry (group A without the three interface keratometries)
  LR      L2-regularised logistic regression (C = 1), median imputation, standardised features,
          49 biomarkers
  RF      random forest (300 trees, minimum leaf size 2), median imputation, 49 biomarkers

Imputation and scaling are learned from the training persons of each fold and, for the independent
set, from the whole development set. Paired differences in AUC: A+B+C minus each other model, and
A minus STD8.

usage: python -m classification.compare_classifiers --features features_split.csv --out results/table_s8.json
"""
import argparse, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .chain import (load_eye_tables, arm_columns, lopo, final_model, design, auc_ci, paired_auc_difference,
                    youden, confusion, SEED)

GENERATOR = "mt19937"
STD8 = ["A_pachy_min", "A_pachy_apex", "A_pachy_global", "A_K_ant", "A_K_post", "A_K_oran",
        "A_tp_dis_merkez_mm", "A_pachy_asimetri"]


def logistic_regression():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def random_forest():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=2))


def raw(g, cols):
    """Biomarker matrix with missing values left as NaN (imputed inside the pipeline)."""
    return g[cols].to_numpy(float)


def models(cols):
    """name -> (biomarkers, model constructor or None for the main classifier, design matrix)."""
    return {"A+B+C": (cols, None, design),
            "A": (arm_columns(cols, "A"), None, design),
            "STD8": (STD8, None, design),
            "LR": (cols, logistic_regression, raw),
            "RF": (cols, random_forest, raw)}


def run(features_csv, out_path, boot=2000):
    g, gb, cols = load_eye_tables(features_csv)
    y, yb = g["label"].values, gb["label"].values
    res, oof, ind = {}, {}, {}
    for name, (kol, make, x) in models(cols).items():
        kw = {"x": x} if make is None else {"make": make, "x": x}
        oof[name] = lopo(g, kol, **kw)
        ind[name] = final_model(g, kol, **kw).predict_proba(x(gb, kol))[:, 1]
        thr = youden(y, oof[name])
        a_dev, ci_dev = auc_ci(g, oof[name], boot, GENERATOR); a_ind, ci_ind = auc_ci(gb, ind[name], boot, GENERATOR)
        res[name] = {"n_features": len(kol), "threshold": thr,
                     "dev_auc": a_dev, "dev_ci": ci_dev, "dev_youden": confusion(y, oof[name], thr),
                     "ind_auc": a_ind, "ind_ci": ci_ind, "ind_carried": confusion(yb, ind[name], thr)}
        print("%-6s development %.3f [%.3f-%.3f]  independent %.3f [%.3f-%.3f]  threshold %.3f"
              % (name, a_dev, *ci_dev, a_ind, *ci_ind, thr), flush=True)
    res["comparisons"] = {}
    for a1, a0 in [("A+B+C", "STD8"), ("A", "STD8"), ("A+B+C", "LR"), ("A+B+C", "RF")]:
        res["comparisons"]["%s - %s" % (a1, a0)] = {"dev": paired_auc_difference(g, oof[a1], oof[a0], boot, GENERATOR),
                                                    "ind": paired_auc_difference(gb, ind[a1], ind[a0], boot, GENERATOR)}
    for k, v in res["comparisons"].items():
        print("%-13s development %+.3f [%+.3f; %+.3f] | independent %+.3f [%+.3f; %+.3f]"
              % (k, v["dev"][0], *v["dev"][1], v["ind"][0], *v["ind"][1]))
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run(a.features, a.out, a.boot)
