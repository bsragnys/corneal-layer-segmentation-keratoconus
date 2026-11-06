"""Agreement coefficients of the reliability sub-study (Section 2.4; Table 2; Table S3).

Cohen's kappa (pixel level, every displaced pixel counts as a disagreement), Fleiss' kappa for
several raters, and the spatially tolerant fuzzy kappa for two raters (improved fuzzy kappa, IFK-2)
adapted from the comparison of categorical maps: each pixel has a fuzzy membership to the classes of
the other mask that decays with distance (Gaussian, sigma = 2 pixels), neighbouring layers receive
a similarity of 0.5, and the expected agreement follows the IFK formulation with the
distribution-based expectation of the minimum (E_min). All coefficients are computed in the region
of interest (ROI): the central +-5 mm band labelled in both masks, background and unlabelled pixels
excluded. Per-image, per-pair coefficients are averaged (simple mean); confidence intervals are
obtained by person-clustered bootstrap.

Mask coding: 2 tear film, 3 epithelium, 4 Bowman's layer, 5 stroma, 6 Descemet-endothelium;
0 background, 7 unlabelled.
"""
from itertools import combinations
import numpy as np
from scipy.ndimage import distance_transform_edt

SIGMA = 2.0          # pixels
HALF_MM = 5.0        # half width of the central band (mm)
NEIGHBOUR = 0.5      # similarity between adjacent layers
LABELS = (2, 3, 4, 5, 6)
MIN_PIXELS = 15


def fgauss(d):
    return np.exp(-(d ** 2) / (2 * SIGMA ** 2))


def build_M(layers, neighbour=NEIGHBOUR):
    layers = np.array(sorted(int(l) for l in layers)); idx = {c: i for i, c in enumerate(layers)}
    M = np.zeros((len(layers), len(layers)))
    for i, ci in enumerate(layers):
        for j, cj in enumerate(layers):
            M[i, j] = 1.0 if ci == cj else (neighbour if abs(ci - cj) == 1 else 0.0)
    return layers, idx, M


def membership(mask, layers, idx, M, ROI, f=fgauss):
    fd = {}
    for i in layers:
        v = (mask == i) & ROI
        fd[i] = f(distance_transform_edt(~v)) if v.any() else np.zeros(mask.shape)
    m = {}
    for cj in layers:
        j = idx[cj]; best = np.zeros(mask.shape)
        for ci in layers:
            w = M[idx[ci], j]
            if w > 0:
                np.maximum(best, w * fd[ci], out=best)
        m[cj] = best
    return m


def E_min(samples):
    samples = [np.asarray(s, float) for s in samples if len(s) > 0]
    if not samples:
        return np.nan
    vals = np.unique(np.concatenate(samples)); vals = vals[vals > 0]
    xs = np.concatenate([[0.0], vals]); E = 0.0
    for k in range(1, len(xs)):
        pr = 1.0
        for s in samples:
            pr *= float((s >= xs[k]).mean())
        E += (xs[k] - xs[k - 1]) * pr
    return E


def ifk2_PE(A, B, mA, mB, ROI, layers):
    """Observed (P) and expected (E) fuzzy agreement of two masks (IFK-2)."""
    aB = np.zeros(A.shape); bA = np.zeros(A.shape)
    for l in layers:
        aB[B == l] = mA[l][B == l]; bA[A == l] = mB[l][A == l]
    P = float(np.minimum(aB, bA)[ROI].mean())
    Ar = A[ROI]; Br = B[ROI]
    piA = {i: float((Ar == i).mean()) for i in layers}; piB = {j: float((Br == j).mean()) for j in layers}
    E = 0.0
    for i in layers:
        if piA[i] == 0:
            continue
        sA = (A == i) & ROI
        for j in layers:
            if piB[j] == 0:
                continue
            sB = (B == j) & ROI
            e = E_min([mA[j][sA], mB[i][sB]])
            if not np.isnan(e):
                E += piA[i] * piB[j] * e
    return P, E


def cohen_PE(A, B, ROI, layers):
    a = A[ROI]; b = B[ROI]
    Po = float((a == b).mean())
    pa = np.array([(a == c).mean() for c in layers]); pb = np.array([(b == c).mean() for c in layers])
    return Po, float((pa * pb).sum())


def fleiss_PE(masks, ROI, layers):
    n = len(masks); labs = np.stack([m[ROI] for m in masks], 1); N = labs.shape[0]
    cnt = np.zeros((N, len(layers)))
    for j, c in enumerate(layers):
        cnt[:, j] = (labs == c).sum(1)
    Pbar = float((((cnt ** 2).sum(1) - n) / (n * (n - 1))).mean())
    pj = cnt.sum(0) / (N * n)
    return Pbar, float((pj ** 2).sum())


def kappa(P, E):
    return (P - E) / (1 - E) if (not np.isnan(P) and not np.isnan(E) and (1 - E) > 1e-12) else np.nan


def roi_band(shape, lateral_spacing_mm, half_mm=HALF_MM):
    """Central band of +-half_mm around the image centre (columns), all rows."""
    H, W = shape
    half = int(round(half_mm / lateral_spacing_mm))
    roi = np.zeros(shape, dtype=bool)
    c = W // 2
    roi[:, max(0, c - half):min(W, c + half + 1)] = True
    return roi


def pair_coefficients(A, B, band):
    """Cohen's kappa, fuzzy kappa (IFK-2) and the number of ROI pixels for one pair of masks."""
    roi = band.copy()
    for m in (A, B):
        roi &= np.isin(m, LABELS)
    if roi.sum() < MIN_PIXELS:
        return None
    lay = np.sort(np.unique(np.concatenate([A[roi], B[roi]]))); lay = lay[lay > 0]
    if len(lay) == 0:
        return None
    layers, idx, M = build_M(lay)
    mA = membership(A, layers, idx, M, roi); mB = membership(B, layers, idx, M, roi)
    P, E = ifk2_PE(A, B, mA, mB, roi, layers); Po, Pe = cohen_PE(A, B, roi, layers)
    return {"cohen": kappa(Po, Pe), "fuzzy": kappa(P, E), "n_pixels": int(roi.sum())}


def group_coefficients(masks, band):
    """Mean pairwise Cohen's and fuzzy kappa and Fleiss' kappa for several masks of one image."""
    roi = band.copy()
    for m in masks:
        roi &= np.isin(m, LABELS)
    if roi.sum() < MIN_PIXELS:
        return None
    lay = np.sort(np.unique(np.concatenate([m[roi] for m in masks]))); lay = lay[lay > 0]
    layers, idx, M = build_M(lay)
    mem = [membership(m, layers, idx, M, roi) for m in masks]
    fuzzy = []; cohen = []
    for a, b in combinations(range(len(masks)), 2):
        P, E = ifk2_PE(masks[a], masks[b], mem[a], mem[b], roi, layers); fuzzy.append(kappa(P, E))
        Po, Pe = cohen_PE(masks[a], masks[b], roi, layers); cohen.append(kappa(Po, Pe))
    Pf, Ef = fleiss_PE(masks, roi, layers)
    return {"cohen_mean_pairwise": float(np.nanmean(cohen)), "fuzzy_mean_pairwise": float(np.nanmean(fuzzy)),
            "fleiss": kappa(Pf, Ef), "n_pixels": int(roi.sum())}


def person_clustered_ci(values, persons, boot=5000, seed=42):
    """Percentile CI of the mean of per-image (or per-pair) values, resampling persons."""
    values = np.asarray(values, float); persons = np.asarray(persons)
    rng = np.random.default_rng(seed); uniq = np.unique(persons); out = []
    for _ in range(boot):
        sel = rng.choice(uniq, len(uniq), replace=True)
        v = np.concatenate([values[persons == q] for q in sel])
        out.append(np.nanmean(v))
    return float(np.nanmean(values)), (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))
