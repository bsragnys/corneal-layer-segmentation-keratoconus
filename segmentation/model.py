"""Four-level U-Net with deep supervision (Section 2.5, Figure 2B).

Each level: two 3 x 3 convolutions, each followed by batch normalisation and ReLU.
Encoder levels 48, 96, 192, 384 channels; bottleneck 768; 2 x 2 max pooling and 2 x 2 transposed
convolutions. Input 640 x 640 x 3 (grey level + two coordinate channels); output: six-class logits
at 640 x 640. Two deep-supervision heads (decoder levels d3 at 160 x 160 and d2 at 320 x 320,
upsampled to the input size) are used only in training. 17,463,378 parameters with base width 48.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 6
BASE_CHANNELS = 48


class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, ic=3, oc=NUM_CLASSES, c=BASE_CHANNELS):
        super().__init__()
        self.e1 = DoubleConv(ic, c);        self.p1 = nn.MaxPool2d(2)
        self.e2 = DoubleConv(c, c * 2);     self.p2 = nn.MaxPool2d(2)
        self.e3 = DoubleConv(c * 2, c * 4); self.p3 = nn.MaxPool2d(2)
        self.e4 = DoubleConv(c * 4, c * 8); self.p4 = nn.MaxPool2d(2)
        self.b = DoubleConv(c * 8, c * 16)
        self.u4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2); self.d4 = DoubleConv(c * 16, c * 8)
        self.u3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2);  self.d3 = DoubleConv(c * 8, c * 4)
        self.u2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2);  self.d2 = DoubleConv(c * 4, c * 2)
        self.u1 = nn.ConvTranspose2d(c * 2, c, 2, 2);      self.d1 = DoubleConv(c * 2, c)
        self.out_main = nn.Conv2d(c, oc, 1)
        self.out_d3 = nn.Conv2d(c * 4, oc, 1)
        self.out_d2 = nn.Conv2d(c * 2, oc, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2)); e4 = self.e4(self.p3(e3))
        b = self.b(self.p4(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        out = self.out_main(d1)
        if self.training:
            ds1 = F.interpolate(self.out_d3(d3), size=x.shape[2:], mode="bilinear", align_corners=False)
            ds2 = F.interpolate(self.out_d2(d2), size=x.shape[2:], mode="bilinear", align_corners=False)
            return out, ds1, ds2
        return out


def build_model(device=None, weights=None):
    model = UNet(ic=3, oc=NUM_CLASSES, c=BASE_CHANNELS)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False))
    if device is not None:
        model = model.to(device)
    return model
