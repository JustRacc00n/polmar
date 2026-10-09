"""
loss.py — Funkcja straty dla korekcji per piksel w przestrzeni polarnej

Trzy składowe (wagi: l1=0.60, mse=0.15, zero=0.20, range=0.05):

1. Rekonstrukcja (L1 + MSE)
   Dopasowanie predykowanego c(θ,r) do c_target(θ,r).
   c_target jest ucięty do >= 0 (dataset.py), więc uczymy się tylko
   usuwania jasnych artefaktów — ciemnych pasm nie dotykamy.

2. Regularyzacja zera (waga: 0.20)
   Kąty bez metalu powinny mieć c(θ,r) ≈ 0 dla wszystkich r.
   Silna regularyzacja wymusza sparsity — model "strzela" tylko w
   kierunkach smugi, wszędzie indziej predykuje blisko zera.
   Zapobiega globalnemu przyciemnieniu.

3. Regularyzacja zakresu HU
   Po odjęciu c(θ,r) wartości pikseli powinny mieścić się
   w zakresie biologicznym [-1000, +1900] HU.
   Zakresy wg skali Hounsfielda:
     -1000 HU — powietrze (dolna granica)
     +1900 HU — kość kortykalna, maks. (górna granica biologiczna)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import HU_MIN, HU_MAX, HU_BIO_LO, HU_BIO_HI

# Granice biologiczne znormalizowane do [0, 1]
# norm = (HU - HU_MIN) / (HU_MAX - HU_MIN), zakres = 4000
NORM_BIO_LO = (HU_BIO_LO - HU_MIN) / (HU_MAX - HU_MIN)  # 0.000 (powietrze)
NORM_BIO_HI = (HU_BIO_HI - HU_MIN) / (HU_MAX - HU_MIN)  # 0.725 (kość kortykalna)


class ScalarCorrectionLoss(nn.Module):
    def __init__(
        self,
        l1_w:    float = 0.60,
        mse_w:   float = 0.15,
        zero_w:  float = 0.20,
        range_w: float = 0.05,
    ):
        super().__init__()
        self.l1_w    = l1_w
        self.mse_w   = mse_w
        self.zero_w  = zero_w
        self.range_w = range_w

    def forward(
        self,
        pred_c:     torch.Tensor,  # [B, N_a, N_r]  predykowana mapa korekcji
        c_target:   torch.Tensor,  # [B, N_a, N_r]  docelowa mapa korekcji
        angle_mask: torch.Tensor,  # [B, N_a]        1 = kąt ze smugą artefaktu
        art_polar:  torch.Tensor,  # [B, N_a, N_r]  obraz z artefaktem (znorm.)
        polar_mask: torch.Tensor,  # [B, N_a, N_r]  1 = piksel metalu w biegunowej
    ):
        # 1. Rekonstrukcja — per piksel
        diff = pred_c - c_target          # [B, N_a, N_r]
        l1   = torch.abs(diff).mean()
        mse  = (diff ** 2).mean()

        # 2. Regularyzacja zera — kąty bez smugi: całe promienie c ≈ 0
        # angle_mask [B, N_a] → [B, N_a, 1] broadcastuje na [B, N_a, N_r]
        non_metal_ang = (1.0 - angle_mask).unsqueeze(-1)          # [B, N_a, 1]
        zero_reg = (pred_c.abs() * non_metal_ang).mean()

        # 3. Regularyzacja zakresu HU — kąty ze smugą, POZA pikselami metalu.
        # NAPRAWKA: poprzednio mnożono tylko przez angle_mask_3d — piksele metalu
        # (HU > 3000, znorm. ≈ 1.0 > NORM_BIO_HI = 0.725) generowały gradient
        # sprzeczny z L1/MSE (L_range chciał podnieść pred_c, L1 ją zerowała).
        # Teraz: (1 - polar_mask) wyklucza metal z regularyzacji zakresu.
        angle_mask_3d = angle_mask.unsqueeze(-1)                   # [B, N_a, 1]
        non_metal_3d  = 1.0 - polar_mask                          # [B, N_a, N_r]
        corrected  = art_polar - pred_c                            # [B, N_a, N_r]
        below      = F.relu(NORM_BIO_LO - corrected)
        above      = F.relu(corrected - NORM_BIO_HI)
        range_mask = angle_mask_3d * non_metal_3d                 # [B, N_a, N_r]
        range_reg  = ((below + above) * range_mask).mean()

        total = (self.l1_w    * l1
               + self.mse_w   * mse
               + self.zero_w  * zero_reg
               + self.range_w * range_reg)

        return total, {
            'l1':    l1.item(),
            'mse':   mse.item(),
            'zero':  zero_reg.item(),
            'range': range_reg.item(),
        }
