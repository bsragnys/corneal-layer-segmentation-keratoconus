"""Sensitivity of the classification to a +-1-pixel shift of the Bowman's layer boundaries
(Section 2.9; Section 3.4; Table S7; Figure S2).

In the independent set, with the reproduced chain and the carried thresholds, the Bowman's layer
thickness biomarkers (rings 1-5, apex, global) are shifted by one axial pixel (9.243 um multiplied by
the format-specific scale factor), either alone, together with a compensating change in the stroma
(posterior boundary shift) or in the epithelium (anterior boundary shift), or with a random sign per
eye in 1000 draws; the composite indices are recomputed and the models are not retrained.

usage: python -m classification.sensitivity_bowman features_split.csv results.json
"""
import pandas as pd, numpy as np, json, sys, os
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
CSV = sys.argv[1]
PX_MM = 0.009243                      # axial sampling interval of the training grid (mm)
_F = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "scale_factors.json")))["factors"]
if any(v is None for v in _F.values()):
    sys.exit("config/scale_factors.json: the format-specific scale factors are not distributed with this "
             "repository; enter the values (available from the authors on request) before running this script.")
CARPAN = {"narrow": _F["narrow"], "wide": _F["wide"], "dar": _F["narrow"], "genis": _F["wide"]}   # format-specific scale factors (config/scale_factors.json)
d = pd.read_csv(CSV)
d["pid2"] = d["pid"].astype(str)
FE = [c for c in d.columns if c[:2] in ("A_", "B_", "C_")]
assert len(FE) == 49, len(FE)
ST8 = ["A_K_oran","A_pachy_min","B_epi_z4","A_tp_dis_merkez_mm","B_bow_z1","B_desc_z5","C_Kant_epi_z2","C_str_apex_pachy"]
BOW = ["B_bow_z1","B_bow_z2","B_bow_z3","B_bow_z4","B_bow_z5","B_bow_apex","B_bow_global"]
STR = [c.replace("bow","str") for c in BOW]; EPI = [c.replace("bow","epi") for c in BOW]

def goz(df):
    g = df.groupby("goz_id")
    return g[FE].mean(), g[["label","kume","pid2","format"]].first()

def perturb(df, mode, sign, eye_sign=None):
    p = df.copy()
    s = p["format"].map(CARPAN).values
    sg = p["goz_id"].map(eye_sign).values if eye_sign is not None else np.full(len(p), float(sign))
    delta = sg * PX_MM * s
    for c in BOW: p[c] = p[c] + delta
    if mode == "posterior":
        for c in STR: p[c] = p[c] - delta
    if mode == "anterior":
        for c in EPI: p[c] = p[c] - delta
    def upd(col, new):
        m = p[col].notna(); p.loc[m, col] = new[m]
    # composite indices recomputed at the image level
    upd("C_bow_apex_global", p.B_bow_apex / p.B_bow_global)
    upd("C_str_apex_pachy", p.B_str_apex / p.A_pachy_apex)
    upd("C_epi_apex_pachy", p.B_epi_apex / p.A_pachy_apex)
    upd("C_epi_remodel", (p.B_epi_global - p.B_epi_apex) / p.B_epi_apex)
    upd("C_Kant_epi_apex", p.A_K_ant * p.B_epi_apex / s)
    upd("C_Kant_epi_z2", p.A_K_ant * p.B_epi_z2 / s)
    return p

def model(n, lr):
    return xgb.XGBClassifier(n_estimators=n, max_depth=3, learning_rate=lr, subsample=0.9,
                             colsample_bytree=0.9, random_state=42, n_jobs=2, eval_metric="logloss")

def lopo(X, y, pid, feats, n, lr):
    oof = np.zeros(len(y))
    for q in np.unique(pid):
        te = pid == q
        m = model(n, lr).fit(X.loc[~te, feats].fillna(-999), y[~te])
        oof[te] = m.predict_proba(X.loc[te, feats].fillna(-999))[:, 1]
    return oof

def youden(y, p):
    f, t, th = roc_curve(y, p); return th[np.argmax(t - f)]

def sesp(y, p, th):
    yp = (p >= th).astype(int)
    return float(((yp == 1) & (y == 1)).sum() / (y == 1).sum()), float(((yp == 0) & (y == 0)).sum() / (y == 0).sum())

dev = d[d.kume == "gelistirme"]; ind = d[d.kume == "bagimsiz"]
Xd, md = goz(dev); yd = md.label.values; pd_ = md.pid2.values
Xi, mi = goz(ind); yi = mi.label.values; pi = mi.pid2.values
rng = np.random.default_rng(2026)
upid = np.unique(pi); idx_by = {q: np.where(pi == q)[0] for q in upid}
B = [np.concatenate([idx_by[q] for q in rng.choice(upid, len(upid), replace=True)]) for _ in range(2000)]

def boot_auc(y, p):
    v = [roc_auc_score(y[b], p[b]) for b in B if 0 < y[b].sum() < len(b)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def boot_diff(y, p0, p1):
    v = [roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]) for b in B if 0 < y[b].sum() < len(b)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

out = {}
for ad, feats, n, lr in [("A+B+C", FE, 200, 0.1), ("STABLE-8", ST8, 300, 0.05)]:
    oof = lopo(Xd, yd, pd_, feats, n, lr)
    th = youden(yd, oof)
    m = model(n, lr).fit(Xd[feats].fillna(-999), yd)
    p0 = m.predict_proba(Xi[feats].fillna(-999))[:, 1]
    r = {"dev_lopo_auc": float(roc_auc_score(yd, oof)), "threshold": float(th),
         "ind_auc": float(roc_auc_score(yi, p0)), "ind_auc_ci": boot_auc(yi, p0), "ind_sens_spec": sesp(yi, p0, th), "scenarios": {}}
    for mode in ["bowman", "posterior", "anterior"]:
        for sign in [+1, -1]:
            Xp, _ = goz(perturb(ind, mode, sign))
            p1 = m.predict_proba(Xp.loc[Xi.index, feats].fillna(-999))[:, 1]
            r["scenarios"]["%s_%+d" % (mode, sign)] = {"auc": float(roc_auc_score(yi, p1)), "auc_ci": boot_auc(yi, p1), "dauc_ci": boot_diff(yi, p0, p1),
                                                     "dauc": float(roc_auc_score(yi, p1) - r["ind_auc"]), "sens_spec": sesp(yi, p1, th)}
    aucs = []; ses = []; sps = []
    eyes = ind.goz_id.unique()
    for k in range(1000):
        es = dict(zip(eyes, rng.choice([-1.0, 1.0], len(eyes))))
        Xp, _ = goz(perturb(ind, "bowman", 0, eye_sign=es))
        p1 = m.predict_proba(Xp.loc[Xi.index, feats].fillna(-999))[:, 1]
        aucs.append(roc_auc_score(yi, p1)); s_, p_ = sesp(yi, p1, th); ses.append(s_); sps.append(p_)
    r["random_shift"] = {"auc_median": float(np.median(aucs)), "auc_2.5_97.5": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
                            "sens_median": float(np.median(ses)), "spec_median": float(np.median(sps))}
    out[ad] = r
bw = Xi.join(mi)
out["bowman_apex_thickness_um"] = {f + "_" + str(l): float(bw[(bw.format == f) & (bw.label == l)]["B_bow_apex"].median() * 1000) for f in sorted(bw.format.unique()) for l in [0, 1]}
out["shift_um"] = {f: PX_MM * 1000 * CARPAN[f] for f in sorted(ind.format.unique())}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
