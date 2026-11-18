"""Column-wise comparison of the model and reference masks for the two thinnest layers in the
held-out test split (Sections 2.9 and 3.2; Table S11).

For Bowman's layer and the Descemet-endothelium complex, in every column of an image in which the
layer is present in both masks and is bounded by labelled pixels in the reference (the pixels just
above and just below the layer are not the ignore index): the axial distance between the two masks
at the anterior and at the posterior boundary of the layer and the difference in layer thickness
(model - reference). The values are averaged over the valid columns of each image and then over
the images; they are given in pixels, in micrometres (axial sampling interval of each image) and,
for the thickness error, as a percentage of the reference thickness. Confidence intervals:
person-clustered bootstrap (2000 resamples).

Both masks are on the same 640 x 640 grid (training class indices; 255 = not labelled), so no
resampling is involved.

usage: python -m segmentation.boundary_columnwise --pred-dir DIR --data-dir DIR --stems split_images.json --split test
(predictions saved as <stem>_pred.npy; reference <stem>_mask.npy and <stem>_meta.npy from prepare_data.py)
"""
import os, json, argparse
import numpy as np
from .preprocessing import IGNORE_INDEX

LAYERS = {"Bowman": 3, "Descemet-endothelium": 5}       # training class indices
MEASURES = ["abs_anterior", "abs_posterior", "signed_anterior", "signed_posterior", "signed_thickness",
            "abs_thickness", "reference_thickness", "model_thickness"]


def column_measures(ref, pred, c):
    """Means over the valid columns of one image (pixels); None if there is no valid column."""
    H, W = ref.shape
    r = ref == c; p = pred == c
    both = r.any(axis=0) & p.any(axis=0)
    top_r = np.argmax(r, axis=0); bot_r = H - np.argmax(r[::-1], axis=0)        # bottom = last row + 1
    top_p = np.argmax(p, axis=0); bot_p = H - np.argmax(p[::-1], axis=0)
    j = np.arange(W)
    above = ref[np.clip(top_r - 1, 0, H - 1), j]; below = ref[np.clip(bot_r, 0, H - 1), j]
    valid = both & (top_r > 0) & (bot_r < H) & (above != IGNORE_INDEX) & (below != IGNORE_INDEX)
    if not valid.any():
        return None
    th_r = r.sum(axis=0)[valid].astype(float); th_p = p.sum(axis=0)[valid].astype(float)
    d_top = (top_p[valid] - top_r[valid]).astype(float); d_bot = (bot_p[valid] - bot_r[valid]).astype(float)
    return {"n_columns": int(valid.sum()), "abs_anterior": np.abs(d_top).mean(), "abs_posterior": np.abs(d_bot).mean(),
            "signed_anterior": d_top.mean(), "signed_posterior": d_bot.mean(),
            "signed_thickness": (th_p - th_r).mean(), "abs_thickness": np.abs(th_p - th_r).mean(),
            "reference_thickness": th_r.mean(), "model_thickness": th_p.mean()}


def clustered_ci(persons, values, boot=2000, seed=42):
    rs = np.random.RandomState(seed)
    persons = np.asarray(persons); values = np.asarray(values, float)
    u = np.unique(persons); groups = {k: np.where(persons == k)[0] for k in u}
    est = [values[np.concatenate([groups[k] for k in rs.choice(u, len(u), replace=True)])].mean() for _ in range(boot)]
    return [float(v) for v in np.percentile(est, [2.5, 97.5])]


def evaluate(pred_dir, data_dir, stems, boot=2000):
    rows = []
    for s in stems:
        ref = np.load(os.path.join(data_dir, f"{s}_mask.npy")).astype(np.int16)
        pred = np.load(os.path.join(pred_dir, f"{s}_pred.npy")).astype(np.int16)
        if ref.shape != pred.shape:
            continue
        um = float(np.load(os.path.join(data_dir, f"{s}_meta.npy")).astype(float)[1]) * 1000.0     # axial um per pixel
        for name, c in LAYERS.items():
            m = column_measures(ref, pred, c)
            if m is not None:
                rows.append({"person": s.split("_")[0], "layer": name, "um": um, **m})
    out = {}
    for name in LAYERS:
        x = [t for t in rows if t["layer"] == name]
        if not x:
            continue
        persons = [t["person"] for t in x]; um = np.array([t["um"] for t in x])
        o = {"n_images": len(x), "n_persons": len(set(persons)),
             "median_valid_columns": float(np.median([t["n_columns"] for t in x]))}
        for key in MEASURES:
            v = np.array([t[key] for t in x], float)
            o[key + "_px"] = [float(v.mean())] + clustered_ci(persons, v, boot)
            o[key + "_um"] = [float((v * um).mean())] + clustered_ci(persons, v * um, boot)
        pct = 100 * np.array([t["signed_thickness"] / t["reference_thickness"] for t in x])
        o["signed_thickness_percent"] = [float(pct.mean())] + clustered_ci(persons, pct, boot)
        out[name] = o
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True); ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stems", required=True, help="split_images.json written by segmentation.train")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    stems = json.load(open(a.stems))[a.split]
    print(json.dumps(evaluate(a.pred_dir, a.data_dir, stems), indent=1, default=float))
