"""Layer-wise segmentation metrics in the held-out test split (Section 3.2; Table S6).

For each image and each class present in the reference, pixels not labelled in the reference
(ignore index) are excluded. Dice, normalised surface Dice (NSD, tolerance 2 pixels), 95th-percentile
Hausdorff distance (HD95) and average symmetric surface distance (ASSD) are computed on the
640 x 640 grid; distances are isotropic pixel distances on that grid. Point estimates are means
over the images in which the class is labelled; confidence intervals for Dice are obtained by
person-clustered bootstrap (persons resampled with replacement).

usage: python -m segmentation.evaluate --pred-dir DIR --data-dir DIR --stems split_images.json --split test
"""
import os, json, argparse
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from .preprocessing import IGNORE_INDEX, CLASS_NAMES


def _surface(b):
    return b & ~binary_erosion(b)


def _surface_distances(a, b):
    """Distances from the surface of a to the surface of b and vice versa (pixels)."""
    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return None, None
    da = distance_transform_edt(~sb)[sa]   # from a's surface to b's surface
    db = distance_transform_edt(~sa)[sb]
    return da, db


def class_metrics(pred, ref, c, tol=2.0):
    valid = ref != IGNORE_INDEX
    p = (pred == c) & valid; r = (ref == c) & valid
    if not r.any():
        return None
    inter = (p & r).sum(); dice = 2 * inter / (p.sum() + r.sum() + 1e-8)
    da, db = _surface_distances(p, r)
    if da is None:
        return {"dice": float(dice), "nsd": np.nan, "hd95": np.nan, "assd": np.nan}
    nsd = ((da <= tol).sum() + (db <= tol).sum()) / (len(da) + len(db))
    hd95 = max(np.percentile(da, 95), np.percentile(db, 95))
    assd = (da.sum() + db.sum()) / (len(da) + len(db))
    return {"dice": float(dice), "nsd": float(nsd), "hd95": float(hd95), "assd": float(assd)}


def evaluate(pred_dir, data_dir, stems, boot=1000, seed=42):
    rows = []
    for s in stems:
        pred = np.load(os.path.join(pred_dir, f"{s}_pred.npy")); ref = np.load(os.path.join(data_dir, f"{s}_mask.npy"))
        for c in range(1, 6):
            m = class_metrics(pred, ref, c)
            if m is not None:
                rows.append({"stem": s, "person": s.split("_")[0], "class": c, **m})
    import pandas as pd
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    persons = df["person"].unique()
    out = {}
    for c in range(1, 6):
        sub = df[df["class"] == c]
        vals = []
        for _ in range(boot):
            sel = rng.choice(persons, len(persons), replace=True)
            v = pd.concat([sub[sub["person"] == q] for q in sel])["dice"]
            if len(v):
                vals.append(v.mean())
        out[CLASS_NAMES[c]] = {"dice": sub["dice"].mean(), "dice_ci": [np.percentile(vals, 2.5), np.percentile(vals, 97.5)],
                               "nsd": sub["nsd"].mean(), "hd95": sub["hd95"].mean(), "assd": sub["assd"].mean(), "n": int(len(sub))}
    # macro-Dice: mean over the five classes per image, then over images
    img_macro = df.groupby("stem")["dice"].mean()
    out["macro_dice"] = {"mean": float(img_macro.mean()), "n_images": int(len(img_macro))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True); ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stems", required=True, help="split_images.json written by segmentation.train")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    stems = json.load(open(a.stems))[a.split]
    print(json.dumps(evaluate(a.pred_dir, a.data_dir, stems), indent=1, default=float))
