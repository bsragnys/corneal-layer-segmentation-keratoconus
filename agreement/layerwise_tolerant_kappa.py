"""Layer-wise model-reference agreement in persons unseen in training (Section 2.4; Table S4):
one-versus-rest spatially tolerant kappa per layer with fixed class-specific tolerances
(sigma in pixels: tear film 1.0, epithelium 2.0, Bowman's layer 1.5, stroma 3.0,
Descemet-endothelium 1.5). Reference masks may contain only a subset of the layers; a layer absent
from the reference is not scored.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .kappa import E_min, kappa

CLASS_SIGMA = {2: 1.0, 3: 2.0, 4: 1.5, 5: 3.0, 6: 1.5}     # original label coding
NAMES = {2: "Tear film", 3: "Epithelium", 4: "Bowman", 5: "Stroma", 6: "Descemet-endothelium"}


def _f(d, s):
    return np.exp(-(d ** 2) / (2 * s ** 2))


def layerwise(reference, prediction, roi, scale=1.0):
    scores = {}
    M = np.array([[1.0, 0.5], [0.5, 1.0]])
    for lbl, name in NAMES.items():
        sigma = CLASS_SIGMA[lbl] * scale
        if not np.any((reference == lbl) & roi):
            scores[name] = np.nan; continue
        gt = (reference == lbl).astype(int); pr = (prediction == lbl).astype(int)
        mu_gt, mu_pr = {}, {}
        for c in (0, 1):
            inv = ~((gt == c) & roi); inv[~roi] = True
            mu_gt[c] = _f(distance_transform_edt(inv), sigma); mu_gt[c][~roi] = 0.
            inv2 = ~((pr == c) & roi); inv2[~roi] = True
            mu_pr[c] = _f(distance_transform_edt(inv2), sigma); mu_pr[c][~roi] = 0.
        mAB = {cj: np.maximum(mu_gt[0] * M[0, cj], mu_gt[1] * M[1, cj]) for cj in (0, 1)}
        mBA = {ci: np.maximum(mu_pr[0] * M[ci, 0], mu_pr[1] * M[ci, 1]) for ci in (0, 1)}
        simA = (pr == 0) * mAB[0] + (pr == 1) * mAB[1]
        simB = (gt == 0) * mBA[0] + (gt == 1) * mBA[1]
        sim = np.minimum(simA, simB); sim = np.where(roi, sim, np.nan)
        P = float(np.nanmean(sim))
        E = 0.0
        piA = {i: float((gt[roi] == i).mean()) for i in (0, 1)}; piB = {j: float((pr[roi] == j).mean()) for j in (0, 1)}
        for i in (0, 1):
            for j in (0, 1):
                if piA[i] == 0 or piB[j] == 0:
                    continue
                e = E_min([mAB[j][(gt == i) & roi], mBA[i][(pr == j) & roi]])
                if not np.isnan(e):
                    E += piA[i] * piB[j] * e
        scores[name] = kappa(P, E)
    return scores
