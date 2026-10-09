"""
compute_metrics.py — Inferencja + metryki jakości MAR na zbiorze testowym

Dla każdego obrazu liczy (poza strefą metalu):
  MAE, RMSE, PSNR, SSIM, gradient MAE
  + gain vs. baseline (oryginalny obraz z artefaktem)

Wynik: CSV z per-image metrykami + podsumowanie na stdout

Uruchomienie:
    python3 compute_metrics.py \
        --model-path  /app/CircNet/mar_run_XXXXX/best_model.pth \
        --data-root   /app/data/RPI/body \
        --test-folder body5 \
        --output-dir  /app/CircNet/mar_run_XXXXX/metrics \
        --polar-size  512x1024
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from dataset import (
    load_raw, normalize, denormalize,
    polar_cv2, get_all_metal_centers,
    find_subfolders,
    HU_MIN, HU_MAX, DEFAULT_POLAR_SIZE,
)
from model import ScalarCorrectionNet
from train import parse_polar_size
import torch.nn as nn


# -------------------------------------------------------------------------
# Stara architektura (RadiusEncoder) — do ładowania historycznych checkpointów
# -------------------------------------------------------------------------
class _OldRadiusEncoder(nn.Module):
    def __init__(self, in_ch=2, out_ch=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Conv1d(16, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):
        B, C, N_a, N_r = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N_a, C, N_r)
        x = self.conv(x)
        return x.squeeze(-1).reshape(B, N_a, -1).permute(0, 2, 1)


class _OldScalarCorrectionNet(nn.Module):
    """Kompatybilność wsteczna z checkpointami trenowanymi starą architekturą."""
    def __init__(self, radius_features=32):
        super().__init__()
        from model import AngleUNet1D
        self.radius_enc = _OldRadiusEncoder(in_ch=2, out_ch=radius_features)
        self.angle_unet = AngleUNet1D(in_ch=radius_features, features=(64, 128, 256))
    def forward(self, x):
        feats = self.radius_enc(x)
        return self.angle_unet(feats).squeeze(1)


def infer_architecture(state):
    """Odczytuje parametry architektury bezpośrednio z kształtów wag."""
    # Stary RadiusEncoder?
    old_arch = any(k.startswith('radius_enc.') for k in state)

    # Softplus na głowicy?
    has_softplus = 'angle_unet.head.0.weight' in state

    if not old_arch:
        # radius_features = wyjście ostatniej warstwy proj
        rf = state['radius_proj.proj.3.weight'].shape[0]

        # angle_features = rozmiary encoderów U-Net
        af = []
        i = 0
        while f'angle_unet.encoders.{i}.block.0.weight' in state:
            af.append(state[f'angle_unet.encoders.{i}.block.0.weight'].shape[0])
            i += 1
    else:
        rf = None
        af = None

    return old_arch, has_softplus, rf, tuple(af) if af else None


def load_model(model_path, n_polar, radius_features, device):
    """Ładuje model — automatycznie wykrywa architekturę z kształtów wag."""
    state = torch.load(model_path, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    old_arch, has_softplus, rf, af = infer_architecture(state)

    if old_arch:
        print('[INFO] Architektura: RadiusEncoder (stara)')
        model = _OldScalarCorrectionNet(radius_features=32).to(device)
    else:
        print(f'[INFO] Architektura: RadiusProjector, rf={rf}, af={af}, softplus={has_softplus}')
        model = ScalarCorrectionNet(
            n_radii=n_polar[0],
            radius_features=rf,
            angle_features=af,
        ).to(device)
        if not has_softplus:
            # Checkpoint bez Softplus — podmień głowicę na liniową
            model.angle_unet.head = nn.Conv1d(af[0], 1, 1).to(device)

    model.load_state_dict(state)
    model.eval()
    return model


# -------------------------------------------------------------------------
# Metryki
# -------------------------------------------------------------------------
WIN_MIN, WIN_MAX = -1000.0, 1000.0


def window01(img_hu: np.ndarray) -> np.ndarray:
    return (np.clip(img_hu, WIN_MIN, WIN_MAX) - WIN_MIN) / (WIN_MAX - WIN_MIN)


def masked_mae(a, b, mask):
    return (np.abs(a - b) * mask).sum() / (mask.sum() + 1e-8)


def masked_rmse(a, b, mask):
    return np.sqrt(((a - b) ** 2 * mask).sum() / (mask.sum() + 1e-8))


def masked_psnr(a, b, mask, data_range=1.0):
    mse = (((a - b) ** 2) * mask).sum() / (mask.sum() + 1e-8)
    return 20 * np.log10(data_range) - 10 * np.log10(mse + 1e-12)


def masked_ssim(a, b, mask, data_range=1.0):
    _, ssim_map = ssim(a, b, data_range=data_range, full=True)
    return (ssim_map * mask).sum() / (mask.sum() + 1e-8)


def sobel_mag(img: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


# -------------------------------------------------------------------------
# Inferencja (sekwencyjna per metal, kopiowana z inference.py)
# -------------------------------------------------------------------------
def correct_image(art_norm, msk, model, centers, device, n_polar):
    H, W = art_norm.shape
    n_radii  = n_polar[0]
    n_angles = n_polar[1]
    tissue_mask = (msk < 0.5).astype(np.float32)
    Y, X        = np.ogrid[:H, :W]
    corrected   = art_norm.copy()

    for (cx, cy) in centers:
        art_polar, r_max = polar_cv2(denormalize(corrected), cx, cy, n=n_polar)
        msk_polar, _     = polar_cv2(msk, cx, cy, n=n_polar, is_mask=True)

        art_polar_n = normalize(art_polar)
        msk_b       = (msk_polar > 0.5).astype(np.float32)

        x_in = np.stack([art_polar_n, msk_b], axis=0)
        x_t  = torch.from_numpy(x_in).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_c = model(x_t).squeeze(0).cpu().numpy()  # [N_a, N_r]

        dx = X - cx
        dy = Y - cy
        theta     = np.arctan2(dy, dx)
        theta     = (theta + 2 * np.pi) % (2 * np.pi)
        angle_idx = (theta / (2 * np.pi) * n_angles).astype(int) % n_angles

        r_dist = np.sqrt(dx ** 2 + dy ** 2)
        r_idx  = (r_dist / r_max * n_radii).astype(int).clip(0, n_radii - 1)

        corrected = np.clip(
            corrected - pred_c[angle_idx, r_idx] * tissue_mask,
            0.0, 1.0,
        )

    return corrected


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',      type=str, required=True)
    p.add_argument('--data-root',       type=str, required=True)
    p.add_argument('--test-folder',     type=str, required=True,
                   help='Nazwa folderu testowego, np. body5')
    p.add_argument('--output-dir',      type=str, required=True)
    p.add_argument('--polar-size',      type=str, default='512x1024')
    p.add_argument('--radius-features', type=int, default=64)
    p.add_argument('--device',          type=str, default='cuda')
    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    n_polar = parse_polar_size(args.polar_size)

    # Załaduj model (auto-detekcja architektury)
    model = load_model(args.model_path, n_polar, args.radius_features, device)
    print(f'Model: {args.model_path}')

    # Zbierz trójki (art, tgt, msk)
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idx_pat = re.compile(r'img(\d+)')
    root = find_subfolders(data_root / args.test_folder)
    baseline_dir = root / 'Baseline'
    target_dir   = root / 'Target'
    mask_dir     = root / 'Mask'

    art_map = {int(idx_pat.search(f.name).group(1)): f
               for f in sorted(baseline_dir.glob('*img*.raw'))
               if idx_pat.search(f.name)}
    tgt_map = {int(idx_pat.search(f.name).group(1)): f
               for f in sorted(target_dir.glob('*img*.raw'))
               if idx_pat.search(f.name)}
    msk_map = {int(re.search(r'(\d+)', f.name).group(1)): f
               for f in sorted(mask_dir.glob('*mask*.raw'))}

    common = sorted(set(art_map) & set(tgt_map) & set(msk_map))
    print(f'Obrazów testowych: {len(common)}  (folder: {args.test_folder})')

    rows = []
    for img_idx in common:
        art = np.clip(load_raw(art_map[img_idx]), HU_MIN, HU_MAX)
        tgt = np.clip(load_raw(tgt_map[img_idx]), HU_MIN, HU_MAX)
        msk = load_raw(msk_map[img_idx])

        art_norm = normalize(art)
        centers  = get_all_metal_centers(msk)

        pred_norm = correct_image(art_norm, msk, model, centers, device, n_polar)

        pred_hu = denormalize(pred_norm)
        metal_mask = (msk > 0.5).astype(np.float32)  # 1 = metal
        outmask    = 1.0 - metal_mask                 # 1 = tkanka (gdzie liczymy metryki)

        # Referencja: czysta tkanka z Target + metal z oryginalnego obrazu
        # Chcemy usunąć smugi, ale metal ma zostać — dlatego w strefie metalu
        # referencją jest sam obraz z artefaktem (nie clean Target bez metalu)
        ref_hu = (1.0 - metal_mask) * tgt + metal_mask * art

        # HU → windowed [0,1] do PSNR/SSIM
        art_w  = window01(art)
        ref_w  = window01(ref_hu)
        pred_w = window01(pred_hu)

        mae_art  = masked_mae(art,    ref_hu, outmask)
        mae_pred = masked_mae(pred_hu, ref_hu, outmask)

        rmse_art  = masked_rmse(art,    ref_hu, outmask)
        rmse_pred = masked_rmse(pred_hu, ref_hu, outmask)

        psnr_art  = masked_psnr(art_w,  ref_w, outmask)
        psnr_pred = masked_psnr(pred_w, ref_w, outmask)

        ssim_art  = masked_ssim(art_w,  ref_w, outmask)
        ssim_pred = masked_ssim(pred_w, ref_w, outmask)

        g_ref  = sobel_mag(ref_w)
        g_art  = sobel_mag(art_w)
        g_pred = sobel_mag(pred_w)
        grad_mae_art  = masked_mae(g_art,  g_ref, outmask)
        grad_mae_pred = masked_mae(g_pred, g_ref, outmask)

        rows.append({
            'iid':           img_idx,
            'mae_metal':     mae_art,
            'mae_pred':      mae_pred,
            'rmse_metal':    rmse_art,
            'rmse_pred':     rmse_pred,
            'psnr_metal':    psnr_art,
            'psnr_pred':     psnr_pred,
            'ssim_metal':    ssim_art,
            'ssim_pred':     ssim_pred,
            'grad_mae_metal': grad_mae_art,
            'grad_mae_pred': grad_mae_pred,
        })
        print(f'  #{img_idx:4d}  MAE: {mae_art:.2f} → {mae_pred:.2f} HU  '
              f'PSNR: {psnr_art:.2f} → {psnr_pred:.2f} dB  '
              f'SSIM: {ssim_art:.3f} → {ssim_pred:.3f}')

    df = pd.DataFrame(rows)
    df['mae_gain']  = df['mae_metal']  - df['mae_pred']
    df['rmse_gain'] = df['rmse_metal'] - df['rmse_pred']
    df['psnr_gain'] = df['psnr_pred']  - df['psnr_metal']
    df['ssim_gain'] = df['ssim_pred']  - df['ssim_metal']
    df['grad_gain'] = df['grad_mae_metal'] - df['grad_mae_pred']

    csv_path = output_dir / f'metrics_{args.test_folder}.csv'
    df.to_csv(csv_path, index=False)

    print(f'\n=== Średnia poprawa vs. oryginalny obraz z artefaktem ===')
    print(f'  MAE   : {df["mae_metal"].mean():.2f} → {df["mae_pred"].mean():.2f} HU  '
          f'(gain: {df["mae_gain"].mean():.2f})')
    print(f'  RMSE  : {df["rmse_metal"].mean():.2f} → {df["rmse_pred"].mean():.2f} HU  '
          f'(gain: {df["rmse_gain"].mean():.2f})')
    print(f'  PSNR  : {df["psnr_metal"].mean():.2f} → {df["psnr_pred"].mean():.2f} dB  '
          f'(gain: {df["psnr_gain"].mean():.2f})')
    print(f'  SSIM  : {df["ssim_metal"].mean():.3f} → {df["ssim_pred"].mean():.3f}  '
          f'(gain: {df["ssim_gain"].mean():.4f})')
    print(f'  GradMAE: {df["grad_mae_metal"].mean():.4f} → {df["grad_mae_pred"].mean():.4f}  '
          f'(gain: {df["grad_gain"].mean():.4f})')
    # --- Summary JSON (kompatybilny z collect_metrics.py) ---
    # Nazwy kolumn: mae_metal → mae_art, psnr_metal → psnr_art itd.
    summary = {
        'test_folder': args.test_folder,
        'n_images':    len(df),
        'prior':       'none (CircNet only)',
        'metrics': {
            'mae_art':   {'mean': float(df['mae_metal'].mean()),    'std': float(df['mae_metal'].std())},
            'mae_pred':  {'mean': float(df['mae_pred'].mean()),     'std': float(df['mae_pred'].std())},
            'rmse_art':  {'mean': float(df['rmse_metal'].mean()),   'std': float(df['rmse_metal'].std())},
            'rmse_pred': {'mean': float(df['rmse_pred'].mean()),    'std': float(df['rmse_pred'].std())},
            'psnr_art':  {'mean': float(df['psnr_metal'].mean()),   'std': float(df['psnr_metal'].std())},
            'psnr_pred': {'mean': float(df['psnr_pred'].mean()),    'std': float(df['psnr_pred'].std())},
            'ssim_art':  {'mean': float(df['ssim_metal'].mean()),   'std': float(df['ssim_metal'].std())},
            'ssim_pred': {'mean': float(df['ssim_pred'].mean()),    'std': float(df['ssim_pred'].std())},
            'grad_art':  {'mean': float(df['grad_mae_metal'].mean()),'std': float(df['grad_mae_metal'].std())},
            'grad_pred': {'mean': float(df['grad_mae_pred'].mean()), 'std': float(df['grad_mae_pred'].std())},
        },
    }
    json_path = output_dir / 'summary.json'
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)

    print(f'\nMetryki zapisane: {csv_path}')
    print(f'Summary JSON:     {json_path}')


if __name__ == '__main__':
    main()
