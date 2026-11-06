"""Inference with the locked network.

usage: python -m segmentation.infer --weights best_unet.pth --dicom FILE [--out mask.npy]
Returns the 640 x 640 mask in the original label coding (2 tear film, 3 epithelium, 4 Bowman's
layer, 5 stroma, 6 Descemet-endothelium) together with the pixel spacing on the network grid.
"""
import argparse
import numpy as np
import torch
from .model import build_model
from .preprocessing import read_dicom, preprocess_image, network_input, inverse_remap_mask


def predict_mask(model, img640, device):
    t = torch.from_numpy(network_input(img640)[None]).float().to(device)
    with torch.no_grad():
        return model(t).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


def segment_dicom(model, path, device):
    img, px, py = read_dicom(path, resample_axial=True)
    W = img.shape[1]
    img640 = preprocess_image(img)
    pred = predict_mask(model, img640, device)
    px640 = px * W / 1024.0        # lateral spacing on the 1024-column grid (mm)
    return inverse_remap_mask(pred), px640, py


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--dicom", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, a.weights).eval()
    mask, px, py = segment_dicom(model, a.dicom, device)
    print("mask", mask.shape, "lateral spacing (mm)", px, "axial spacing (mm)", py)
    if a.out:
        np.save(a.out, mask)
