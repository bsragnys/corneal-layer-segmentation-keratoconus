"""Loss terms used in training (Section 2.5, Figure 2C).

Effective loss = class-wise focal Tversky + Dice + online hard-example mining (OHEM, top 25% of
pixel cross-entropy), summed with equal weights. The training code also contained a
signed-distance boundary term (weight 0.1) and a Bowman's layer continuity term (weight 0.05,
applied to full images only). A code review after training showed that these two terms made no
meaningful contribution: the continuity term is computed on a thresholded map and is therefore
not differentiable (no gradient), and the boundary term, whose distance map is normalised and
signed as implemented here, contributed a loss of the order of 2e-4, negligible compared with the
Dice term. They are kept so that the published code matches the code that was run.

Deep supervision: the same loss is computed on the two auxiliary outputs with weights
main 1.0, ds1 0.4, ds2 0.2, and the weighted sum is divided by 1.6.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

NUM_CLASSES = 6
IGNORE_INDEX = 255
BOWMAN_TRAIN_LBL = 3
OHEM_KEEP_RATIO = 0.25
# class-wise focal Tversky parameters (alpha = false-positive penalty, beta = false-negative penalty, gamma)
CLASS_TVERSKY = {1: (0.3, 0.7, 1.5), 2: (0.3, 0.7, 1.5), 3: (0.7, 0.3, 1.5), 4: (0.4, 0.6, 1.3), 5: (0.2, 0.8, 1.5)}
DS_WEIGHTS = {"main": 1.0, "ds1": 0.4, "ds2": 0.2}


class PerClassFocalTverskyLoss(nn.Module):
    def __init__(self, params=CLASS_TVERSKY, nc=NUM_CLASSES, ii=IGNORE_INDEX, s=1e-6):
        super().__init__(); self.p = params; self.nc = nc; self.ii = ii; self.s = s

    def forward(self, logits, target):
        vm = target != self.ii; st = target.clone(); st[~vm] = 0
        p = torch.softmax(logits, 1) * vm.unsqueeze(1).float()
        toh = F.one_hot(st, self.nc).permute(0, 3, 1, 2).float() * vm.unsqueeze(1).float()
        losses = []
        for c in range(1, self.nc):
            a, b, g = self.p.get(c, (0.3, 0.7, 1.5))
            pp = p[:, c].reshape(-1); tt = toh[:, c].reshape(-1)
            TP = (pp * tt).sum(); FP = (pp * (1 - tt)).sum(); FN = ((1 - pp) * tt).sum()
            losses.append(((1 - (TP + self.s) / (TP + a * FP + b * FN + self.s))) ** g)
        return torch.mean(torch.stack(losses))


class DiceLoss(nn.Module):
    def __init__(self, nc=NUM_CLASSES, ii=IGNORE_INDEX, s=1e-6):
        super().__init__(); self.nc = nc; self.ii = ii; self.s = s

    def forward(self, logits, target):
        vm = target != self.ii; st = target.clone(); st[~vm] = 0
        p = torch.softmax(logits, 1) * vm.unsqueeze(1).float()
        toh = F.one_hot(st, self.nc).permute(0, 3, 1, 2).float() * vm.unsqueeze(1).float()
        losses = [1 - (2 * (p[:, c] * toh[:, c]).sum() + self.s) / (p[:, c].sum() + toh[:, c].sum() + self.s)
                  for c in range(1, self.nc)]
        return torch.mean(torch.stack(losses))


class OHEMLoss(nn.Module):
    def __init__(self, kr=OHEM_KEEP_RATIO, ii=IGNORE_INDEX):
        super().__init__(); self.kr = kr; self.ii = ii

    def forward(self, logits, target):
        lm = F.cross_entropy(logits, target, ignore_index=self.ii, reduction="none")
        valid = lm[target != self.ii]
        if valid.numel() == 0:
            return lm.mean()
        k = max(1, int(valid.numel() * self.kr))
        return torch.topk(valid, k, largest=True, sorted=False)[0].mean()


class BoundaryLoss(nn.Module):
    """Signed-distance boundary term as implemented in the training run (see module docstring)."""
    def __init__(self, nc=NUM_CLASSES, ii=IGNORE_INDEX):
        super().__init__(); self.nc = nc; self.ii = ii

    @staticmethod
    def _dm(m):
        if not np.any(m):
            return np.zeros_like(m, dtype=np.float32)
        s = distance_transform_edt(m) - distance_transform_edt(~m)
        return (s / (np.abs(s).max() + 1e-8)).astype(np.float32)

    def forward(self, logits, target):
        vm = target != self.ii; st = target.clone(); st[~vm] = 0
        p = torch.softmax(logits, 1); B = p.shape[0]
        loss = torch.tensor(0., device=logits.device); n = 0
        for b in range(B):
            t = st[b].cpu().numpy(); v = vm[b].cpu().numpy()
            for c in range(1, self.nc):
                mc = (t == c) & v
                if not np.any(mc):
                    continue
                loss += (p[b, c] * torch.from_numpy(self._dm(mc)).to(logits.device)).mean(); n += 1
        return loss / max(n, 1)


class ConnectivityLoss(nn.Module):
    """Continuity term counting breaks in the thresholded Bowman's layer prediction (non-differentiable)."""
    def __init__(self, bc=BOWMAN_TRAIN_LBL, thr=0.3, ii=IGNORE_INDEX):
        super().__init__(); self.bc = bc; self.thr = thr; self.ii = ii

    @staticmethod
    def _rr(b2d):
        total = active = 0
        for row in b2d:
            if not np.any(row):
                continue
            total += int(np.sum(np.diff(np.concatenate([[False], row, [False]]).astype(int)) == 1)); active += 1
        return total / max(active, 1)

    def forward(self, logits, target):
        p = torch.softmax(logits, 1); B = p.shape[0]; pen = 0.
        for b in range(B):
            pb = (p[b, self.bc] > self.thr).cpu().numpy()
            tb = (target[b].cpu().numpy() == self.bc) & (target[b].cpu().numpy() != self.ii)
            if not np.any(tb):
                continue
            pen += max(0., self._rr(pb) - self._rr(tb))
        return torch.tensor(pen / max(B, 1), dtype=torch.float32, device=logits.device)


ftversky_fn = PerClassFocalTverskyLoss()
dice_fn = DiceLoss()
ohem_fn = OHEMLoss()
boundary_fn = BoundaryLoss()
conn_fn = ConnectivityLoss()


def compute_loss(logits, mask, with_conn=True):
    loss = ftversky_fn(logits, mask) + dice_fn(logits, mask) + ohem_fn(logits, mask) + 0.1 * boundary_fn(logits, mask)
    if with_conn:
        loss = loss + 0.05 * conn_fn(logits, mask)
    return loss


def compute_loss_ds(outputs, mask, with_conn=True):
    main, ds1, ds2 = outputs; w = DS_WEIGHTS
    return (w["main"] * compute_loss(main, mask, with_conn)
            + w["ds1"] * compute_loss(ds1, mask, False)
            + w["ds2"] * compute_loss(ds2, mask, False)) / sum(w.values())


def seg_metrics(logits, target, nc=NUM_CLASSES, ii=IGNORE_INDEX):
    """Per-class Dice and macro-Dice (mean over the five non-background classes) for model selection."""
    pred = torch.argmax(logits, 1); valid = target != ii; res = {}; dv = []
    for c in range(1, nc):
        pc = (pred == c) & valid; tc = (target == c) & valid
        i = (pc & tc).sum().item(); ps = pc.sum().item(); ts = tc.sum().item()
        d = (2 * i) / (ps + ts + 1e-8) if (ps + ts) > 0 else np.nan
        res[f"dice_{c}"] = d
        if not np.isnan(d):
            dv.append(d)
    res["macro_dice"] = float(np.mean(dv)) if dv else np.nan
    return res
