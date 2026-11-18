"""Refraction correction of sections exported without refraction correction (Section 2.3).

In an uncorrected section the axial dimension is the optical path length: distances inside the
cornea appear larger by the group refractive index, and structures below the inclined anterior
surface are displaced laterally. The correction is made by ray tracing before measurement:

1. the image is resampled to a lateral interval of 15.738 um (the lateral grid of the training
   images) and segmented with the locked model to locate the anterior corneal surface;
2. the surface profile is smoothed with a Savitzky-Golay filter (41 pixels, third order) to obtain
   its local slope;
3. telecentric scanning is assumed: every A-scan enters vertically, is refracted at the anterior
   surface according to Snell's law and travels in a straight line through a medium with the group
   refractive index n_g = 1.389; a pixel at an optical path length l below the surface belongs to
   the tissue position at a distance l / n_g along the refracted ray;
4. the image is resampled to these positions by linear interpolation; the region above the anterior
   surface is left unchanged;
5. the corrected image is segmented again with the same locked model, and all measurements are
   made from that mask (measurement/build_feature_table.py).
"""
import numpy as np
from scipy.ndimage import map_coordinates, median_filter
from scipy.signal import savgol_filter
from skimage.transform import resize as sk_resize
from segmentation.preprocessing import CROP_SIZE, PX_REF, cornea_top_row

N_GROUP = 1.389            # group refractive index of the cornea
SG_WINDOW, SG_ORDER = 41, 3


def resample_lateral(img, px, px_ref=PX_REF):
    """Bring the lateral sampling interval to px_ref (mm); the axial direction is unchanged."""
    H, W = img.shape
    W2 = max(8, int(round(W * px / px_ref)))
    return sk_resize(img.astype(np.float32), (H, W2), order=1, preserve_range=True,
                     anti_aliasing=(W2 < W)).astype(np.float32)


def crop_window(img, r0=None, size=CROP_SIZE):
    """size x size network window of an image on the lateral reference grid: rows from r0 (by
    default about 60 rows above the top of the cornea), columns centred as in the standard
    preprocessing; columns outside the image are zero. Returns (window, r0, c0)."""
    H, W2 = img.shape
    if r0 is None:
        r0 = cornea_top_row(img, th=size)
    x = img[r0:r0 + size, :]
    if x.shape[0] < size:
        x = np.pad(x, ((0, size - x.shape[0]), (0, 0)), constant_values=0)
    centre = W2 / 2.0 + 2.0 * (W2 / 1024.0)
    c0 = int(round(centre - size / 2))
    window = np.zeros((size, size), np.float32)
    k0 = max(0, c0); k1 = min(W2, c0 + size)
    window[:, k0 - c0:k1 - c0] = x[:, k0:k1]
    return window, r0, c0


def anterior_surface(pred, r0, c0, n_columns, size=CROP_SIZE):
    """Row of the anterior corneal surface in every column of the laterally resampled image, from
    the mask of the first pass (class indices 1-5; 0 background).

    A fourth-degree polynomial is fitted to the first labelled row of each column with three rounds
    of outlier rejection (3 x 1.4826 x MAD + 0.5 pixel). Where the median-filtered (9 columns)
    boundary of the mask lies within that distance of the fit, the boundary itself is kept, so that
    local irregularities of the surface are preserved; elsewhere the fit is used. Returns None when
    fewer than 50 columns contain the cornea."""
    m = (pred > 0) & (pred <= 5)
    present = m.any(0); top = np.argmax(m, 0).astype(float)
    cols = np.arange(size) + c0
    ok = present & (cols >= 0) & (cols < n_columns)
    if ok.sum() < 50:
        return None
    x, y = cols[ok].astype(float), top[ok] + r0
    xx, yy = x.copy(), y.copy()
    for _ in range(3):
        p = np.polyfit(xx, yy, 4); r = yy - np.polyval(p, xx)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 0.5
        k = np.abs(r) < 3 * s; xx, yy = xx[k], yy[k]
    a = np.polyval(p, np.arange(n_columns, dtype=float))
    boundary = np.full(n_columns, np.nan)
    boundary[x.astype(int)] = median_filter(y, size=9, mode="nearest")
    good = np.isfinite(boundary) & (np.abs(boundary - a) < 3 * s)
    a[good] = boundary[good]
    return a


def correct_refraction(img, surface_row, py, px=PX_REF, n_group=N_GROUP, iterations=5):
    """Ray-traced image of an uncorrected section.

    img: image on the lateral reference grid (row interval py mm of optical path, column interval
    px mm); surface_row: row of the anterior surface in every column (anterior_surface).
    For every output pixel below the surface, the entry column of the refracted ray that passes
    through it is found by fixed-point iteration; with d the path length in tissue from the entry
    point, the output pixel is sampled from the input image at the entry column and at the row
    surface_row + n_group * d / py (linear interpolation). Pixels above the surface are copied."""
    H, W = img.shape
    j0 = np.arange(W, dtype=np.float64)
    smooth = savgol_filter(surface_row, SG_WINDOW, SG_ORDER, mode="interp")
    za = surface_row * py                                   # depth of the surface, mm
    s = np.gradient(smooth * py, px)                        # slope of the surface, dz/dx
    q = np.sqrt(1 + s * s); c1 = 1 / q; eta = 1 / n_group   # cos(incidence); ratio of the indices
    c2 = np.sqrt(np.clip(1 - eta * eta * (1 - c1 * c1), 0, 1)); k = eta * c1 - c2
    tx = k * s / q; tz = eta - k / q                        # direction of the refracted ray
    norm = np.sqrt(tx * tx + tz * tz); tx, tz = tx / norm, tz / norm
    R, C = np.mgrid[0:H, 0:W].astype(np.float64); zp = R * py
    below = zp > np.interp(C, j0, za)
    jj = C[below].copy(); z = zp[below]
    for _ in range(iterations):
        d = (z - np.interp(jj, j0, za)) / np.interp(jj, j0, tz)
        jj = C[below] - d * np.interp(jj, j0, tx) / px
    d = (z - np.interp(jj, j0, za)) / np.interp(jj, j0, tz)
    rr = np.interp(jj, j0, surface_row) + n_group * d / py
    SR, SC = R.copy(), C.copy(); SR[below] = rr; SC[below] = jj
    return map_coordinates(img, [SR, SC], order=1, mode="constant", cval=0.0).astype(np.float32)
