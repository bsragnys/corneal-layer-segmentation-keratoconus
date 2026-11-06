"""Convert DICOM sections and their label masks into the processed arrays used for training.

usage: python -m segmentation.prepare_data --dicom-dir DIR --mask-dir DIR --out-dir DIR
Masks are .npy arrays in the original label coding (0 background, 2 tear film, 3 epithelium,
4 Bowman's layer, 5 stroma, 6 Descemet-endothelium, 7 unlabelled); mask file name
<stem>__va_mask.npy, <stem>_mask.npy or <stem>.npy.
"""
import os, argparse
from pathlib import Path
import numpy as np
import pydicom
from .preprocessing import preprocess_image, preprocess_mask, remap_mask, get_pixel_spacing


def is_dicom(p):
    try:
        pydicom.dcmread(p, stop_before_pixels=True, force=True); return True
    except Exception:
        return False


def find_dicom_files(folder):
    paths = []
    for r, _, files in os.walk(folder):
        rel = os.path.relpath(r, folder)
        if rel != "." and any(pp.startswith("_") for pp in rel.split(os.sep)):
            continue
        for fn in sorted(files):
            fp = os.path.join(r, fn)
            if fn.lower().endswith(".dcm"):
                paths.append(fp)
            elif "." not in fn and is_dicom(fp):
                paths.append(fp)
    return sorted(paths)


def find_mask(mask_dir, stem):
    for s in [f"{stem}__va_mask.npy", f"{stem}_mask.npy", f"{stem}.npy"]:
        p = os.path.join(mask_dir, s)
        if os.path.exists(p):
            return p
    return None


def process_and_save(dcm_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ok = miss = 0
    for dp in find_dicom_files(dcm_dir):
        stem = Path(dp).stem
        oi = os.path.join(out_dir, f"{stem}_img.npy"); om = os.path.join(out_dir, f"{stem}_mask.npy")
        if os.path.exists(oi) and os.path.exists(om):
            continue
        mp = find_mask(mask_dir, stem)
        if mp is None:
            miss += 1; continue
        ds = pydicom.dcmread(dp, force=True)
        img = ds.pixel_array.astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        img = img * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        if "MONOCHROME1" in str(getattr(ds, "PhotometricInterpretation", "")).upper():
            img = img.max() - img
        scale = 1024. / img.shape[1]
        mask = np.load(mp)
        if mask.ndim == 3:
            mask = mask[0]
        img_p, r0 = preprocess_image(img, return_r0=True)
        np.save(oi, img_p.astype(np.float32))
        np.save(om, remap_mask(preprocess_mask(mask, r0=r0)))
        px, py = get_pixel_spacing(ds)
        np.save(os.path.join(out_dir, f"{stem}_meta.npy"), np.array([px / scale, py]))
        ok += 1
    return ok, miss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-dir", required=True); ap.add_argument("--mask-dir", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    print(process_and_save(a.dicom_dir, a.mask_dir, a.out_dir))
