"""Layer measurements, ring zones and the 49 candidate biomarkers (Section 2.6, Figure 3, Table S2).

Input: a 640 x 640 mask in the original label coding (2 tear film, 3 epithelium, 4 Bowman's layer,
5 stroma, 6 Descemet-endothelium; the implementation below works on the training grid coding
1-5 after subtracting one, see `from_original`), the lateral spacing px and the axial spacing py
on the network grid (mm). Output: a dictionary of image-level biomarkers.

Measurement unit: the image column. In each column the anterior boundary is the top of the
epithelium and the posterior boundary the bottom of the Descemet-endothelium band. Thickness is
measured perpendicular to the anterior surface (vertical extent x cos theta, theta = local slope
of the median-filtered anterior surface). Column-level quality filter: Descemet-endothelium band
present; total perpendicular thickness within 0.5-1.5 x the image median and within 0.25-1.10 mm;
images with fewer than 150 valid columns are not measured. Ring zones z1-z5 are defined by the
horizontal distance from the vertex of a conic fitted to the anterior surface; the apex window is
+-0.5 mm around the thinnest point; asymmetry = |mean(left half) - mean(right half)| with the halves
split at the vertex. Keratometry K = (1.3375 - 1) / R from the apical radius of a conic fitted to
each surface or interface with a robust (MAD) refit.
"""
import numpy as np
from scipy import ndimage
from scipy.signal import medfilt
from scipy.optimize import curve_fit

MIN_VALID_COLUMNS = 150
N_KERATO = 1.3375
APEX_HALF_MM = 0.5
LAYERS = {"epi": 2, "bow": 3, "str": 4, "desc": 5}     # training grid coding (0 background, 1 tear film)


def from_original(mask_original):
    """Original label coding (2..6) -> coding used here (1..5); 0 and 7 -> 0."""
    m = mask_original.astype(np.int16) - 1
    m[(mask_original == 0) | (mask_original == 7)] = 0
    return m.astype(np.uint8)


def conic_off(x, R, Q, x0, z0):
    xx = x - x0
    return z0 + xx ** 2 / (R * (1 + np.sqrt(np.maximum(1 - (Q + 1) * (xx / R) ** 2, 1e-10))))


def fit_conic(x, z):
    """Apical radius R (mm) and asphericity Q of a conic fitted to (x, z); robust refit after
    removing points whose residual exceeds 3 x 1.4826 x MAD. Returns (nan, nan) with < 50 points."""
    if len(x) < 50:
        return np.nan, np.nan
    xm = x[np.argmin(z)]; xx, zz = x - xm, z - z.min()
    try:
        p, _ = curve_fit(conic_off, xx, zz, p0=[7.8, -.26, 0., 0.], bounds=([3., -5., -3., -1.], [15., 2., 3., 1.]), maxfev=10000)
        res = zz - conic_off(xx, *p)
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        keep = np.abs(res - np.median(res)) <= max(3 * mad, 1e-6)
        if keep.sum() >= 50 and keep.sum() < len(xx):
            p, _ = curve_fit(conic_off, xx[keep], zz[keep], p0=p, bounds=([3., -5., -3., -1.], [15., 2., 3., 1.]), maxfev=10000)
        return float(p[0]), float(p[1])
    except Exception:
        return np.nan, np.nan


def keratometry(R):
    return (N_KERATO - 1.) / (R / 1000.) if (np.isfinite(R) and R > 0) else np.nan


def extract_features(mask, px, py):
    """Image-level biomarkers from a mask (coding 1..5) with pixel spacings px, py in mm."""
    m = np.where(mask > 5, 0, mask).astype(np.uint8)
    fg = m > 0
    lab, ncomp = ndimage.label(fg)
    if ncomp == 0:
        return None
    sizes = ndimage.sum(fg, lab, range(1, ncomp + 1))
    m = np.where(lab == (1 + int(np.argmax(sizes))), m, 0)       # largest connected component
    H, W = m.shape
    a_r = np.full(W, np.nan); p_r = np.full(W, np.nan)
    thick_v = {k: np.full(W, np.nan) for k in LAYERS}
    bounds = {"bow_top": np.full(W, np.nan), "str_top": np.full(W, np.nan), "str_bot": np.full(W, np.nan)}
    for c in range(W):
        col = m[:, c]
        e = np.where((col >= 2) & (col <= 5))[0]
        d = np.where(col == 5)[0]
        if e.size == 0 or d.size == 0:
            continue
        a_r[c] = e[0]; p_r[c] = d[-1]
        for k, l in LAYERS.items():
            thick_v[k][c] = float((col == l).sum()) * py
        b3 = np.where(col == 3)[0]
        if b3.size:
            bounds["bow_top"][c] = b3[0]
        s4 = np.where(col == 4)[0]
        if s4.size:
            bounds["str_top"][c] = s4[0]; bounds["str_bot"][c] = s4[-1]
    valid = np.isfinite(a_r)
    if valid.sum() < MIN_VALID_COLUMNS:
        return None
    z_a = a_r * py; z_s = z_a.copy()
    gi = np.where(valid)[0]
    z_s[valid] = medfilt(z_a[valid], 9)
    slope = np.full(W, np.nan); slope[gi] = np.gradient(z_s[gi], gi * px)
    costh = 1.0 / np.sqrt(1.0 + slope ** 2)
    tot = (p_r - a_r) * py * costh                                 # perpendicular total thickness (mm)
    thick = {k: v * costh for k, v in thick_v.items()}
    med = np.nanmedian(tot[valid])
    ok = valid & (tot > 0.5 * med) & (tot < 1.5 * med) & (tot > 0.25) & (tot < 1.1)
    if ok.sum() < MIN_VALID_COLUMNS:
        return None
    cols = np.where(ok)[0]
    # thinnest point: slope-constrained minimum of the median-smoothed total thickness
    steep_ok = ok & (costh >= 0.8)
    cand = np.where(steep_ok)[0]
    if cand.size >= MIN_VALID_COLUMNS:
        inner = cand[(cand > cols[0] + 10) & (cand < cols[-1] - 10)]
        cand = inner if inner.size >= 50 else cand
        sm = medfilt(tot[cand], 7); tp_i = int(cand[int(np.argmin(sm))])
        pachy_min = float(np.min(sm))
    else:
        sm = medfilt(tot[cols], 7); tp_i = int(cols[int(np.argmin(sm))])
        pachy_min = float(np.min(sm))
    # vertex of the conic fitted to the anterior surface = zone centre
    Ra, _ = fit_conic(cols * px, a_r[cols] * py)
    xm = (cols * px)[np.argmin(a_r[cols] * py)]
    xx = cols * px - xm; zz = a_r[cols] * py - (a_r[cols] * py).min()
    try:
        p, _ = curve_fit(conic_off, xx, zz, p0=[7.8, -.26, 0., 0.], bounds=([3., -5., -3., -1.], [15., 2., 3., 1.]), maxfev=10000)
        centre = int(round((xm + p[2]) / px))
    except Exception:
        centre = tp_i                                             # fallback: thinnest point
    if not (cols[0] <= centre <= cols[-1]):
        centre = tp_i
    idx = np.arange(W)
    dist = np.abs(idx - centre) * px                              # horizontal distance from the vertex (mm)
    w_ap = max(3, int(round(APEX_HALF_MM / px)))
    ap = ok & (idx >= tp_i - w_ap) & (idx <= tp_i + w_ap)

    def mean_of(arr, sel):
        v = arr[sel]; v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else np.nan

    f = {"kalite_sutun": int(ok.sum()), "v2_merkez_tp_fark_mm": float(abs(centre - tp_i) * px)}
    Rp, _ = fit_conic(cols * px, p_r[cols] * py)
    Ka, Kp = keratometry(Ra), keratometry(Rp)
    f["A_pachy_min"] = pachy_min
    f["A_pachy_apex"] = mean_of(tot, ap)
    f["A_pachy_global"] = mean_of(tot, ok)
    f["A_K_ant"] = Ka; f["A_K_post"] = Kp
    f["A_K_oran"] = Ka / Kp if (np.isfinite(Ka) and np.isfinite(Kp) and Kp > 0) else np.nan
    f["A_tp_dis_merkez_mm"] = float(abs(tp_i - W / 2) * px)

    def interface_K(arr):
        okk = ok & np.isfinite(arr)
        if okk.sum() < 50:
            return np.nan
        R, _ = fit_conic(idx[okk] * px, arr[okk] * py)
        return keratometry(R)

    f["A_K_bow_ant"] = interface_K(bounds["bow_top"])      # epithelium-Bowman interface
    f["A_K_str_ant"] = interface_K(bounds["str_top"])      # Bowman-stroma interface
    f["A_K_str_post"] = interface_K(bounds["str_bot"])     # stroma-Descemet interface
    for k in LAYERS:
        for zi in range(5):
            f[f"B_{k}_z{zi + 1}"] = mean_of(thick[k], ok & (dist >= zi) & (dist < zi + 1))
        f[f"B_{k}_apex"] = mean_of(thick[k], ap)
        f[f"B_{k}_global"] = mean_of(thick[k], ok)
        left = mean_of(thick[k], ok & (idx < centre)); right = mean_of(thick[k], ok & (idx >= centre))
        f[f"B_{k}_asimetri"] = abs(left - right) if np.isfinite(left) and np.isfinite(right) else np.nan
    tl = mean_of(tot, ok & (idx < centre)); tr = mean_of(tot, ok & (idx >= centre))
    f["A_pachy_asimetri"] = abs(tl - tr) if np.isfinite(tl) and np.isfinite(tr) else np.nan

    def g(name):
        return f.get(name, np.nan)

    f["C_bow_apex_global"] = g("B_bow_apex") / g("B_bow_global") if g("B_bow_global") else np.nan
    f["C_epi_apex_pachy"] = g("B_epi_apex") / g("A_pachy_apex") if g("A_pachy_apex") else np.nan
    f["C_str_apex_pachy"] = g("B_str_apex") / g("A_pachy_apex") if g("A_pachy_apex") else np.nan
    f["C_Kant_epi_apex"] = g("A_K_ant") * g("B_epi_apex")
    f["C_Kant_epi_z2"] = g("A_K_ant") * g("B_epi_z2")
    f["C_epi_remodel"] = ((g("B_epi_global") - g("B_epi_apex")) / g("B_epi_apex")) if g("B_epi_apex") else np.nan
    return f


SCALED_A = ("A_pachy_min", "A_pachy_apex", "A_pachy_global", "A_pachy_asimetri")


def apply_scale_factor(f, fmt, factors):
    """Bring absolute thickness biomarkers to the common (device) scale with the fixed,
    format-specific factor (Section 2.6; config/scale_factors.json). Group C is not scaled."""
    out = dict(f)
    for c in SCALED_A:
        if c in out and out[c] is not None:
            out[c] *= factors[fmt]
    for c in list(out):
        if c.startswith("B_") and out[c] is not None:
            out[c] *= factors[fmt]
    return out


def export_format(n_columns):
    """Format group from the number of columns of the exported DICOM (Table S1)."""
    return "wide" if n_columns >= 1500 else "narrow"
