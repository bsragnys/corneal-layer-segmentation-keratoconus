"""Dataset, augmentation, Bowman-centred patch sampling and the person-level split (Section 2.5).

Expected processed files (produced by prepare_data.py): <stem>_img.npy (640 x 640 float32,
preprocessed), <stem>_mask.npy (640 x 640 uint8, training class indices, 255 = ignore),
<stem>_meta.npy ([lateral spacing on the 1024-column grid, axial spacing] in mm).
The person identifier is the part of the stem before the first underscore.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates, gaussian_filter
from sklearn.model_selection import train_test_split
from .preprocessing import robust_normalize, IGNORE_INDEX

CROP_SIZE = 640
PATCH_SIZE = 256
PATCH_RATIO = 0.40
BOWMAN_TRAIN_LBL = 3


def get_person_id(stem):
    return stem.split("_")[0]


def elastic_deform(img, mask, alpha=60., sigma=8.):
    h, w = img.shape; rng = np.random.default_rng()
    dx = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha
    ix = (np.clip(np.meshgrid(np.arange(w), np.arange(h))[1] + dy, 0, h - 1).ravel(),
          np.clip(np.meshgrid(np.arange(w), np.arange(h))[0] + dx, 0, w - 1).ravel())
    return (map_coordinates(img.astype(np.float32), ix, order=1).reshape(h, w).astype(np.float32),
            map_coordinates(mask.astype(np.float32), ix, order=0).reshape(h, w).astype(np.uint8))


def augment(img, mask):
    """Online augmentation: horizontal flip (p = 0.5), brightness/contrast (factor 0.85-1.15,
    shift +-0.1; p = 0.7), Gaussian noise (sigma = 0.02; p = 0.3), elastic deformation
    (alpha = 60, sigma = 8; p = 0.4)."""
    if random.random() < 0.5:
        img = np.fliplr(img).copy(); mask = np.fliplr(mask).copy()
    if random.random() < 0.7:
        img = np.clip(img * np.float32(random.uniform(0.85, 1.15)) + np.float32(random.uniform(-.1, .1)), 0., 1.)
    if random.random() < 0.3:
        img = np.clip(img + np.random.normal(0, .02, img.shape).astype(np.float32), 0., 1.)
    if random.random() < 0.4:
        img, mask = elastic_deform(img, mask)
    return img.astype(np.float32), mask


def extract_bowman_patch(img, mask):
    """256 x 256 patch centred on Bowman's layer (vertical centre = median row of the layer,
    horizontal centre = median column with a random offset of up to +-64 pixels)."""
    rows = np.where(np.any(mask == BOWMAN_TRAIN_LBL, axis=1))[0]
    yc = int(np.median(rows)) if len(rows) > 0 else img.shape[0] // 3
    y1 = max(0, yc - PATCH_SIZE // 2); y2 = min(img.shape[0], y1 + PATCH_SIZE); y1 = max(0, y2 - PATCH_SIZE)
    cols = np.where(np.any(mask == BOWMAN_TRAIN_LBL, axis=0))[0]
    xc = (int(np.median(cols)) if len(cols) > 0 else img.shape[1] // 2) + random.randint(-PATCH_SIZE // 4, PATCH_SIZE // 4)
    x1 = max(0, xc - PATCH_SIZE // 2); x2 = min(img.shape[1], x1 + PATCH_SIZE); x1 = max(0, x2 - PATCH_SIZE)
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def bowman_weight(mask_path):
    """Sampling weight of an image: 0.1 + 0.9 * min(200 * Bowman fraction, 1)."""
    m = np.load(mask_path); total = np.sum(m != IGNORE_INDEX)
    if total == 0:
        return 0.1
    return float(0.1 + 0.9 * min(np.sum(m == BOWMAN_TRAIN_LBL) / total * 200, 1.))


class CorneaDataset(Dataset):
    def __init__(self, stems, data_dir, training=False, patch_mode=False):
        self.stems = stems; self.training = training; self.pm = patch_mode; self.dd = data_dir
        sz = PATCH_SIZE if patch_mode else CROP_SIZE
        self.xg, self.yg = np.meshgrid(np.linspace(-1, 1, sz, dtype=np.float32), np.linspace(-1, 1, sz, dtype=np.float32))

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        s = self.stems[idx]
        img = np.load(os.path.join(self.dd, f"{s}_img.npy"))
        mask = np.load(os.path.join(self.dd, f"{s}_mask.npy"))
        if self.pm:
            img, mask = extract_bowman_patch(img, mask)
        img = robust_normalize(img)
        if self.training:
            img, mask = augment(img, mask)
        h, w = img.shape
        xg, yg = (self.xg, self.yg) if (h == self.xg.shape[0] and w == self.xg.shape[1]) else \
            np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32))
        return {"image": torch.from_numpy(np.stack([img, xg, yg]).astype(np.float32)),
                "mask": torch.from_numpy(mask.astype(np.int64)).long(),
                "stem": s}


def create_split(stems, fixed_test, forced_train, seed=42):
    """Person-level split of the segmentation pool (config/partition_protocol.md).

    fixed_test: person identifiers set aside as the held-out test split before training;
    forced_train: person identifiers that must be in the training split; the remaining persons are
    divided into training and validation with sklearn.train_test_split(test_size=0.25, random_state=seed).
    No person appears in two splits.
    """
    fixed_test = set(fixed_test); forced_train = set(forced_train)
    persons = sorted(set(get_person_id(s) for s in stems))
    remaining = [p for p in persons if p not in fixed_test and p not in forced_train]
    tr_p, va_p = train_test_split(remaining, test_size=0.25, random_state=seed)
    tr_p = list(tr_p) + list(forced_train & set(persons))
    te_p = list(fixed_test & set(persons))
    tr = [s for s in stems if get_person_id(s) in set(tr_p)]
    va = [s for s in stems if get_person_id(s) in set(va_p)]
    te = [s for s in stems if get_person_id(s) in set(te_p)]
    assert not (set(tr_p) & set(va_p)) and not (set(tr_p) & set(te_p)) and not (set(va_p) & set(te_p))
    return tr, va, te
