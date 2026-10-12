"""SMP 0.5.0 API. Boundary head is training-only; inference stays automatic."""
import torch
from torch import nn
import segmentation_models_pytorch as smp

FAMILIES = ("unet", "unetpp", "linknet", "deeplab", "segformer", "boundary", "consistency", "proposed")


class Segmenter(nn.Module):
    def __init__(self, family, pretrained=True):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"Unknown model: {family}")
        self.family = family
        self.use_boundary = family in ("boundary", "proposed")
        self.use_consistency = family in ("consistency", "proposed")
        kw = dict(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1, activation=None)
        if family in ("unet", "boundary", "consistency", "proposed"):
            self.net = smp.Unet(**kw)
        elif family == "unetpp":
            self.net = smp.UnetPlusPlus(**kw)
        elif family == "linknet":
            self.net = smp.Linknet(**kw)
        elif family == "deeplab":
            self.net = smp.DeepLabV3Plus(**kw)
        else:
            # Official hierarchical MiT-B0 transformer baseline, not a CNN labeled SegFormer.
            kw["encoder_name"] = "mit_b0"
            self.net = smp.Segformer(**kw)
        if self.use_boundary:
            self.edge_head = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, rgb):
        x = (rgb - self.mean) / self.std
        if self.use_boundary:
            features = self.net.encoder(x)
            dec = self.net.decoder(features)
            return self.net.segmentation_head(dec), self.edge_head(dec)
        return self.net(x), None
