"""
inference.py — Aplikacja wytrenowanego modelu do obrazów CT

Dla każdego obrazu wejściowego:
  1. Wykrywa centra metaliczne z maski
  2. Dla każdego centrum predykuje mapę korekcji c(θ, r) [N_angles, N_radii]
  3. Aplikuje korekcję do obrazu kartezjańskiego przez lookup 2D:
       angle_idx = f(x, y, cx, cy)
       r_idx     = (dist(x,y,cx,cy) / r_max * N_r).clip(0, N_r-1)
       corrected(x,y) = art(x,y) − c[angle_idx, r_idx]
  4. Przy wielu metalach korekcje aplikowane sekwencyjnie
  5. Zapisuje wynik jako .raw float32

Uruchomienie:
    python3 inference.py \\
        --model-path  /app/test/mar_run/best_model.pth \\
        --data-root   /app/data/RPI \\
        --test-folders body4 \\
        --output-dir  /app/test/mar_run/corrected \\
        --polar-size  512x1024
"""

import argparse
import json
import re
import numpy as np
import torch
from pathlib import Path

from dataset import (
    load_raw, normalize, denormalize,
    polar_cv2, get_all_metal_centers,
    find_subfolders,
    HU_MIN, HU_MAX, DEFAULT_POLAR_SIZE,
)
from model import ScalarCorrectionNet
from train import parse_polar_size


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='MAR Scalar Correction Inference')
    p.add_argument('--model-path',     type=str, required=True,
                   help='Ścieżka do best_model.pth')
    p.add_argument('--data-root',      type=str, required=True,
                   help='Katalog główny z danymi, np. /app/data/RPI')
    p.add_argument('--test-folders',   type=str, nargs='+', required=True,
                   help='Nazwy folderów testowych, np. body4')
    p.add_argument('--output-dir',     type=str, required=True,
                   help='Katalog wyjściowy na skorygowane obrazy')
    p.add_argument('--polar-size',      type=str, default=None,
                   help='Nadpisuje polar_size z best_model_args.json, np. 512x1024')
    p.add_argument('--radius-features', type=int, default=None,
                   help='Nadpisuje radius_features z best_model_args.json')
    p.add_argument('--device',          type=str, default='cuda')
    return p.parse_args()



def correct_image_with_mask(
    art_norm:   np.ndarray,
    msk:        np.ndarray,
    model:      torch.nn.Module,
    centers:    list,
    device:     torch.device,
    n_polar:    tuple = DEFAULT_POLAR_SIZE,
) -> np.ndarray:
    """
    Sekwencyjnie aplikuje korekcję per piksel dla każdego centrum metalicznego.
    Model predykuje mapę c(θ, r) [N_a, N_r]; każdy piksel kartezjański
    dostaje korekcję przez lookup 2D po (angle_idx, r_idx).

    art_norm : [H, W] znormalizowany obraz z artefaktem [0, 1]
    msk      : [H, W] maska metalu (float32)
    centers  : lista (cx, cy) centrów metalicznych obiektów
    n_polar  : (N_radii, N_angles) — konwencja cv2
    """
    H, W = art_norm.shape
    n_radii  = n_polar[0]
    n_angles = n_polar[1]
    tissue_mask = (msk < 0.5).astype(np.float32)   # [H, W] — stała dla wszystkich metali
    Y, X        = np.ogrid[:H, :W]

    # Korekcja sekwencyjna — każdy metal dostaje obraz już po korekcji poprzedniego.
    corrected = art_norm.copy()

    for (cx, cy) in centers:
        # Wejście z aktualnie skorygowanego obrazu
        art_polar, r_max = polar_cv2(denormalize(corrected), cx, cy, n=n_polar)
        msk_polar, _     = polar_cv2(msk, cx, cy, n=n_polar, is_mask=True)

        art_polar_n = normalize(art_polar)
        msk_b       = (msk_polar > 0.5).astype(np.float32)

        x_in = np.stack([art_polar_n, msk_b], axis=0)  # [2, N_a, N_r]
        x_t  = torch.from_numpy(x_in).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_c = model(x_t).squeeze(0).cpu().numpy()  # [N_a, N_r] >= 0 (Softplus)

        # Mapowanie piksela kartezjańskiego → (angle_idx, r_idx) w polar
        dx = X - cx
        dy = Y - cy
        theta     = np.arctan2(dy, dx)
        theta     = (theta + 2 * np.pi) % (2 * np.pi)
        angle_idx = (theta / (2 * np.pi) * n_angles).astype(int) % n_angles

        # Promień w pikselach → indeks w osi r (skalowanie jak w warpPolar)
        r_dist = np.sqrt(dx ** 2 + dy ** 2)
        r_idx  = (r_dist / r_max * n_radii).astype(int).clip(0, n_radii - 1)

        # Lookup 2D: korekcja zależy od kierunku I odległości od metalu
        corrected = np.clip(
            corrected - pred_c[angle_idx, r_idx] * tissue_mask,
            0.0, 1.0,
        )

    return corrected


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def _load_model_args(model_path: Path) -> dict:
    """Ładuje best_model_args.json z katalogu modelu (jeśli istnieje)."""
    p = model_path.parent / 'best_model_args.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _infer_radius_features(state: dict) -> int:
    """Fallback: wyciąga radius_features z kształtu tensora w state dict."""
    key = 'radius_proj.proj.3.weight'   # idx 3: Linear(hidden → radius_features)
    if key in state:
        return state[key].shape[0]
    raise RuntimeError(
        f"Nie można auto-wykryć radius_features: brak '{key}' w state dict. "
        "Podaj --radius-features ręcznie."
    )


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Auto-wykrywanie parametrów modelu ---
    model_path = Path(args.model_path)
    saved_args = _load_model_args(model_path)

    state = torch.load(model_path, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    if args.radius_features is not None:
        radius_features = args.radius_features
    elif 'radius_features' in saved_args:
        radius_features = saved_args['radius_features']
    else:
        radius_features = _infer_radius_features(state)

    polar_size_str = args.polar_size or saved_args.get('polar_size', '512x1024')
    n_polar = parse_polar_size(polar_size_str)

    # Załaduj model
    model = ScalarCorrectionNet(
        n_radii=n_polar[0],
        radius_features=radius_features,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    if saved_args:
        print(f'(parametry z best_model_args.json: rf={radius_features})')
    print(f'Model załadowany z {model_path}')

    # Buduj listę plików z folderów testowych (tak samo jak dataset)
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zbierz pary (art_path, msk_path) ze wszystkich folderów testowych
    idx_pat = re.compile(r'img(\d+)')
    test_pairs = []
    for folder in args.test_folders:
        root = find_subfolders(data_root / folder)
        baseline_dir = root / 'Baseline'
        mask_dir     = root / 'Mask'

        art_files_d = {
            int(idx_pat.search(f.name).group(1)): f
            for f in sorted(baseline_dir.glob('*img*.raw'))
            if idx_pat.search(f.name)
        }
        msk_files_d = {
            int(re.search(r'(\d+)', f.name).group(1)): f
            for f in sorted(mask_dir.glob('*mask*.raw'))
        }
        for i in sorted(set(art_files_d) & set(msk_files_d)):
            test_pairs.append((art_files_d[i], msk_files_d[i]))

    print(f'Obrazów do przetworzenia: {len(test_pairs)}')

    for art_path, msk_path in test_pairs:
        art_orig = load_raw(art_path)                          # oryginalne HU bez klipowania
        art      = np.clip(art_orig, HU_MIN, HU_MAX)
        msk      = load_raw(msk_path)

        art_norm = normalize(art)
        centers  = get_all_metal_centers(msk)

        corrected_norm = correct_image_with_mask(
            art_norm, msk, model, centers, device, n_polar
        )
        corrected_hu = denormalize(corrected_norm)

        # Przywróć oryginalne wartości HU w strefie metalu.
        # normalize/denormalize ucina wartości do [HU_MIN, HU_MAX], przez co metal
        # (często >3000 HU) byłby zmodyfikowany mimo braku korekcji algorytmem.
        metal_mask   = (msk > 0.5).astype(np.float32)
        corrected_hu = corrected_hu * (1.0 - metal_mask) + art_orig * metal_mask

        out_path = output_dir / art_path.name
        corrected_hu.astype(np.float32).tofile(out_path)
        print(f'  {art_path.name} → {out_path.name}  '
              f'[HU: {corrected_hu.min():.0f} .. {corrected_hu.max():.0f}]')

    print('\nInference zakończony.')


if __name__ == '__main__':
    main()
