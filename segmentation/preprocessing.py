"""Preprocessing of AS-OCT sections (Section 2.3 of the paper).

Steps: (1) the image width is rescaled to 1024 columns, the axial direction is left unchanged;
(2) a 640 x 640 window is cropped: 640 rows beginning about 60 rows above the top of the cornea
and the central 640 of the 1024 columns; (3) intensities are scaled to 0-1 between the 1st and
99th percentiles. Two coordinate channels (horizontal and vertical position in -1..1) are added
to the grey level, giving the 640 x 640 x 3 network input.

In the measurement chain, images whose axial sampling interval differs from the training grid
(9.243 um) by more than 5% are resampled axially to that interval before segmentation.
"""
import numpy as np
import pydicom
from skimage.transform import resize as sk_resize
from scipy.ndimage import zoom

CROP_SIZE = 640
PY_REF = 0.009243          # axial sampling interval of the training grid, mm
IGNORE_INDEX = 255
# original label codes in the masks -> training class indices
ORIGINAL_TO_TRAIN = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: IGNORE_INDEX}
TRAIN_TO_ORIGINAL = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
CLASS_NAMES = {0: "Background", 1: "Tear film", 2: "Epithelium", 3: "Bowman", 4: "Stroma", 5: "Descemet-endothelium"}


def robust_normalize(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    return (np.clip(img, p1, p99) - p1) / (p99 - p1 + 1e-8)


def get_pixel_spacing(ds):
    """Return (lateral, axial) sampling interval in mm from the DICOM header."""
    try:
        sp = ds.PixelSpacing
        return float(sp[1]), float(sp[0])
    except Exception:
        pass
    try:
        sp = ds[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0028, 0x0030].value
        return float(sp[1]), float(sp[0])
    except Exception:
        pass
    return 0.015738, 0.009243


def cornea_top_row(img1024, th=CROP_SIZE, margin=60):
    """First row of the 640-row window: about 60 rows above the first row in which at least
    50 columns are bright (robust to a single-column specular line)."""
    lo, hi = np.percentile(img1024, [1, 99.5])
    n = np.clip((img1024 - lo) / max(hi - lo, 1e-6), 0, 1)
    bright = (n > 0.5).sum(axis=1)
    rows = np.where(bright >= 50)[0]
    r0 = max(0, int(rows[0]) - margin) if len(rows) else 0
    H = img1024.shape[0]
    if r0 + th > H:
        r0 = max(0, H - th)
    return r0


def _width_1024(arr, order):
    return sk_resize(arr, (arr.shape[0], 1024), order=order, preserve_range=True,
                     anti_aliasing=(order == 1)).astype(arr.dtype if order == 0 else np.float32)


def preprocess_image(img, tw=CROP_SIZE, th=CROP_SIZE, return_r0=False):
    img = _width_1024(img.astype(np.float32), order=1)
    r0 = cornea_top_row(img, th=th)
    img = img[r0:r0 + th, :]
    if img.shape[0] < th:
        img = np.pad(img, ((0, th - img.shape[0]), (0, 0)), constant_values=0)
    c0 = max(0, 514 - tw // 2); c1 = min(1024, c0 + tw); c0 = max(0, c1 - tw)
    img = img[:, c0:c1]
    if img.shape[1] < tw:
        img = np.pad(img, ((0, 0), (0, tw - img.shape[1])), constant_values=0)
    img = img.astype(np.float32)
    return (img, r0) if return_r0 else img


def preprocess_mask(mask, r0, tw=CROP_SIZE, th=CROP_SIZE):
    mask = _width_1024(mask, order=0)
    mask = mask[r0:r0 + th, :]
    if mask.shape[0] < th:
        mask = np.pad(mask, ((0, th - mask.shape[0]), (0, 0)), constant_values=IGNORE_INDEX)
    c0 = max(0, 514 - tw // 2); c1 = min(1024, c0 + tw); c0 = max(0, c1 - tw)
    mask = mask[:, c0:c1]
    if mask.shape[1] < tw:
        mask = np.pad(mask, ((0, 0), (0, tw - mask.shape[1])), constant_values=IGNORE_INDEX)
    return mask


def remap_mask(mask):
    out = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for orig, tr in ORIGINAL_TO_TRAIN.items():
        out[mask == orig] = tr
    return out


def inverse_remap_mask(m):
    out = np.zeros_like(m, dtype=np.uint8)
    for t, o in TRAIN_TO_ORIGINAL.items():
        out[m == t] = o
    out[m == IGNORE_INDEX] = 7
    return out


def read_dicom(path, resample_axial=True):
    """Read a DICOM section. Returns (image, lateral spacing mm, axial spacing mm).
    In the measurement chain the axial interval is brought to PY_REF when it differs by > 5%."""
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)
    if img.ndim == 3:
        img = img[0]
    img = img * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    if "MONOCHROME1" in str(getattr(ds, "PhotometricInterpretation", "")).upper():
        img = img.max() - img
    px, py = get_pixel_spacing(ds)
    if resample_axial:
        fy = py / PY_REF
        if abs(fy - 1.0) > 0.05:
            img = zoom(img, (fy, 1.0), order=1)
            py = PY_REF
    return img, px, py


def network_input(img640):
    """Grey level (0-1) plus horizontal and vertical coordinate channels, shape (3, H, W)."""
    n = robust_normalize(img640)
    h, w = n.shape
    xg, yg = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32))
    return np.stack([n, xg, yg]).astype(np.float32)
