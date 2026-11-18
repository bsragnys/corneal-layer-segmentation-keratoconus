"""Performance in the persons of the independent set who took part in no development step
(Sections 2.2, 2.9 and 3.4).

Some persons of the classification cohort had been in the exploratory analyses that preceded the
final cohort, in which the biomarker definitions were developed, or in the segmentation pool. The
script reports these overlaps and, in the independent set, the saved full model (A+B+C) at its
locked threshold in (i) all persons, (ii) the persons who had not been in the exploratory analyses
and (iii) among them, those who were not in the segmentation pool: AUC, sensitivity and specificity
with person-clustered bootstrap intervals.

The two person lists are text files with one person identifier per line; they are study-specific
and are not distributed.

usage: python -m classification.subset_analyses --features features_split.csv --results results/ \
           --exploratory exploratory_persons.txt --segmentation-pool pool_persons.txt \
           --out results/subset_analyses.json
"""
import json, argparse
import numpy as np
from .chain import load_eye_tables, auc_ci, rate_ci, confusion
from .format_analyses import full_model_outputs


def read_list(path):
    return {line.strip() for line in open(path) if line.strip()}


def summary(t, p, thr, boot):
    a, ci = auc_ci(t, p, boot, "pcg64")
    se, se_ci = rate_ci(t, p, thr, 1, boot); sp, sp_ci = rate_ci(t, p, thr, 0, boot)
    return {"persons": int(t["pid"].nunique()), "eyes": len(t), "kc": int((t["label"] == 1).sum()),
            "auc": a, "auc_ci": ci, "sens": se, "sens_ci": se_ci, "spec": sp, "spec_ci": sp_ci,
            **{k: v for k, v in confusion(t["label"].values, p, thr).items() if k in ("TP", "FN", "TN", "FP")}}


def run(features_csv, results_dir, exploratory_txt, pool_txt, out_path, boot=2000):
    g, gb, _ = load_eye_tables(features_csv)
    thr, _, p_ind = full_model_outputs(results_dir, g, gb)
    exploratory, pool = read_list(exploratory_txt), read_list(pool_txt)
    res = {"threshold": thr, "overlap": {}}
    for name, t in (("dev", g), ("ind", gb)):
        persons = set(t["pid"].astype(str))
        res["overlap"][name] = {"persons": len(persons), "segmentation_pool": len(persons & pool),
                                "exploratory": len(persons & exploratory), "either": len(persons & (pool | exploratory))}
    pid = gb["pid"].astype(str)
    subsets = {"all": np.ones(len(gb), bool),
               "not_in_exploratory": (~pid.isin(exploratory)).to_numpy(),
               "not_in_exploratory_nor_pool": (~pid.isin(exploratory) & ~pid.isin(pool)).to_numpy()}
    for name, m in subsets.items():
        res[name] = summary(gb[m].reset_index(drop=True), p_ind[m], thr, boot)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda v: round(float(v), 3)))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--exploratory", required=True); ap.add_argument("--segmentation-pool", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run(a.features, a.results, a.exploratory, a.segmentation_pool, a.out, a.boot)
