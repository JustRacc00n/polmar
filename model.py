"""
model.py — CircNet architecture (ScalarCorrectionNet)

Two-stage polar-domain correction model for metal artifact reduction in CT:
  1. RadiusProjector — projects the full radial profile (N_r values) to per-angle features
  2. AngleUNet1D     — 1D U-Net along the angular axis, outputs correction vector c(θ, r) [N_r]

Input:  [B, 2, N_angles, N_radii]  — artifact image + metal mask in polar space
Output: [B, N_angles, N_radii]     — per-pixel correction map in polar coordinates

Application (in Cartesian space):
    angle_idx = f(x, y, cx, cy)                         # angle bin index
    r_idx     = (dist(x,y,cx,cy) / r_max * N_r).clip() # radius bin index
    corrected(x,y) = art(x,y) − c[angle_idx, r_idx]    # applied outside metal only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadiusProjector(nn.Module):
    """
    Projects the full radial profile to a per-angle feature vector.

    Instead of collapsing N_r values to a scalar (e.g. mean), the network
    learns which parts of the radial profile are informative.

    Input:  [B, 2,       N_angles, N_radii]
    Output: [B, out_ch,  N_angles]

    Steps:
      1. Reshape → [B*N_a, 2*N_r]          (each angle as a flat vector)
      2. Linear(2*N_r → hidden) + ReLU
      3. Linear(hidden → out_ch)
      4. Reshape → [B, out_ch, N_a]
    """
    def __init__(self, n_radii: int = 512, in_ch: int = 2,
                 hidden: int = 256, out_ch: int = 64):
        super().__init__()
        self.in_ch = in_ch
        self.n_radii = n_radii
        self.proj = nn.Sequential(
            nn.Linear(in_ch * n_radii, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_ch),
            nn.LayerNorm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N_a, N_r = x.shape
        # [B, 2, N_a, N_r] → [B*N_a, 2*N_r]  (each angle as a flat vector)
        x = x.permute(0, 2, 1, 3).reshape(B * N_a, C * N_r)
        x = self.proj(x)                                  # [B*N_a, out_ch]
        x = x.reshape(B, N_a, -1).permute(0, 2, 1)       # [B, out_ch, N_a]
        return x


class ConvBlock1D(nn.Module):
    """Double 1D convolutional block with circular padding (the angular axis is periodic)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AngleUNet1D(nn.Module):
    """
    1D U-Net operating along the angular sequence with neighbourhood context.

    Input:  [B, in_ch, N_angles]
    Output: [B, N_angles, N_radii]  — per-pixel correction map in polar coordinates
    """
    def __init__(self, in_ch: int = 64, features: tuple = (128, 256, 512),
                 n_radii: int = 512):
        super().__init__()
        self.n_radii = n_radii
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        self.ups      = nn.ModuleList()
        self.decoders = nn.ModuleList()

        ch = in_ch
        for f in features:
            self.encoders.append(ConvBlock1D(ch, f))
            self.pools.append(nn.MaxPool1d(2))
            ch = f

        self.bottleneck = ConvBlock1D(ch, ch * 2)
        ch = ch * 2

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose1d(ch, f, 2, stride=2))
            self.decoders.append(ConvBlock1D(f * 2, f))
            ch = f

        # Softplus ensures non-negative outputs: log(1 + exp(x)) > 0
        # The model can only subtract — it will never brighten the image.
        # Conv1d(ch, n_radii, 1): predicts N_r correction values per angle.
        self.head = nn.Sequential(
            nn.Conv1d(ch, n_radii, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        out = self.head(x)           # [B, N_r, N_a]
        return out.permute(0, 2, 1)  # [B, N_a, N_r]


class ScalarCorrectionNet(nn.Module):
    """
    CircNet: predicts a per-pixel correction map c(θ, r) in polar space.

    Pipeline:
        [B, 2, N_a, N_r]  →  RadiusProjector  →  [B, 64, N_a]
                          →  AngleUNet1D       →  [B, N_a, N_r]

    Correction application (outside metal only, in Cartesian space):
        r_idx     = (dist(x,y,cx,cy) / r_max * N_r).clip(0, N_r-1)
        angle_idx = angle_bin(x, y, cx, cy)
        corrected(x,y) = art(x,y) − c[angle_idx, r_idx] * (1 − metal_mask(x,y))
    """
    def __init__(self, n_radii: int = 512, radius_features: int = 64,
                 angle_features: tuple = (128, 256, 512)):
        super().__init__()
        self.radius_proj = RadiusProjector(
            n_radii=n_radii, in_ch=2,
            hidden=256, out_ch=radius_features,
        )
        self.angle_unet = AngleUNet1D(
            in_ch=radius_features, features=angle_features, n_radii=n_radii,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, N_a, N_r]  (polar artifact + polar metal mask)
        feats = self.radius_proj(x)      # [B, radius_features, N_a]
        corr  = self.angle_unet(feats)   # [B, N_a, N_r]
        return corr                      # non-negative correction map [B, N_a, N_r]
