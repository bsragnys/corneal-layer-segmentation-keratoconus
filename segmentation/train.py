"""Training and locking of the segmentation network (Section 2.5, Figure 2D).

usage: python -m segmentation.train --data-dir DIR --split config/segmentation_split.json --out-dir DIR

Settings (config/training_config.json): AdamW (lr 1e-4, weight decay 1e-4), cosine schedule over
80 epochs, batch size 4 (8 for Bowman-centred patches), 40% of the iterations on 256 x 256
Bowman-centred patches, weighted sampling by Bowman fraction, mixed precision, gradient clipping
at 1.0, default (Kaiming uniform) initialisation, no pretrained weights. The model is selected by
the macro-Dice in the validation split; training stops after 15 epochs without improvement and the
weights of the best epoch are locked (never modified afterwards).
"""
import os, json, glob, random, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from .model import UNet, NUM_CLASSES, BASE_CHANNELS
from .losses import compute_loss, compute_loss_ds, seg_metrics
from .data import CorneaDataset, create_split, bowman_weight, PATCH_RATIO


def train(data_dir, split_cfg, out_dir, cfg):
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_unet.pth")

    stems = [Path(f).name.replace("_img.npy", "") for f in sorted(glob.glob(os.path.join(data_dir, "*_img.npy")))]
    tr_stems, va_stems, te_stems = create_split(stems, split_cfg["fixed_test_persons"], split_cfg["forced_train_persons"], seed)
    json.dump({"train": tr_stems, "validation": va_stems, "test": te_stems}, open(os.path.join(out_dir, "split_images.json"), "w"), indent=1)
    print(f"train {len(tr_stems)} | validation {len(va_stems)} | test {len(te_stems)} images")

    bs = cfg["batch_size"]
    tw = [bowman_weight(os.path.join(data_dir, f"{s}_mask.npy")) for s in tr_stems]
    sampler = WeightedRandomSampler(tw, len(tr_stems), replacement=True)
    tr_ds_f = CorneaDataset(tr_stems, data_dir, training=True, patch_mode=False)
    tr_ds_p = CorneaDataset(tr_stems, data_dir, training=True, patch_mode=True)
    va_ds = CorneaDataset(va_stems, data_dir, training=False)
    tr_ld_f = DataLoader(tr_ds_f, batch_size=bs, sampler=sampler, num_workers=0, pin_memory=True)
    tr_ld_p = DataLoader(tr_ds_p, batch_size=bs * 2, sampler=sampler, num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(ic=3, oc=NUM_CLASSES, c=BASE_CHANNELS).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val = -1.; no_improve = 0; log = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        n_full = int(len(tr_ld_f) * (1 - PATCH_RATIO)); n_patch = len(tr_ld_f) - n_full
        schedule = random.sample([False] * n_full + [True] * n_patch, n_full + n_patch)
        it_f = iter(tr_ld_f); it_p = iter(tr_ld_p); tloss = 0.
        for use_p in tqdm(schedule, desc=f"epoch {epoch:03d}", leave=False):
            try:
                batch = next(it_p) if use_p else next(it_f)
            except StopIteration:
                it_f = iter(tr_ld_f); it_p = iter(tr_ld_p)
                batch = next(it_p) if use_p else next(it_f)
            img, mask = batch["image"].to(device), batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(img); loss = compute_loss_ds(outputs, mask, with_conn=not use_p)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt); scaler.update(); tloss += loss.item()
        sch.step()
        model.eval(); vl = 0.; vm = []
        with torch.no_grad():
            for b in va_ld:
                img, mask = b["image"].to(device), b["mask"].to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    lg = model(img); vl += compute_loss(lg, mask, with_conn=False).item()
                vm.append(seg_metrics(lg, mask))
        vl /= max(1, len(va_ld))
        dice_vals = [m["macro_dice"] for m in vm if not np.isnan(m["macro_dice"])]
        cur = float(np.mean(dice_vals)) if dice_vals else np.nan
        log.append({"epoch": epoch, "train_loss": tloss / max(1, len(schedule)), "val_loss": vl, "val_macro_dice": cur})
        print(f"epoch {epoch:03d} | train {tloss / max(1, len(schedule)):.4f} | val {vl:.4f} | macro-Dice {cur:.4f}")
        if not np.isnan(cur) and cur > best_val:
            best_val = cur; no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print("early stopping"); break
    json.dump(log, open(os.path.join(out_dir, "training_log.json"), "w"), indent=1)
    print(f"locked model: {best_path} (best validation macro-Dice {best_val:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--split", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config", "training_config.json"))
    a = ap.parse_args()
    train(a.data_dir, json.load(open(a.split)), a.out_dir, json.load(open(a.config)))
