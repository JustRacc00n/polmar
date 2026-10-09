"""
real_circnet_inference.py — CircNet inference na rzeczywistych danych szpitalnych

Wzorzec pliku wejściowego:  metal{NN}_slice{NNNN}_H512_W512.raw
Wzorzec pliku wyjściowego:  metal{NN}_slice{NNNN}_H512_W512_circnet.raw

Maska metalu derywowana z HU > --metal-hu (domyślnie 2500), brak osobnych plików mask.

Uruchomienie:
    python3 real_circnet_inference.py \
        --model-path      /app/CircNet/mar_run_16838558/best_model.pth \
        --input-dir       /app/data/real_raw \
        --output-dir      /app/output/real_circnet \
        --polar-size      512x1024 \
        --radius-features 64
"""

import argparse
import re
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from dataset import (
    load_raw, normalize, denormalize,
    get_all_metal_centers,
    HU_MIN, HU_MAX, DEFAULT_POLAR_SIZE,
)
from model import ScalarCorrectionNet
from train import parse_polar_size
from inference import correct_image_with_mask

# ── Stałe ─────────────────────────────────────────────────────────────────────
REAL_RE      = re.compile(r"^metal(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+\.raw$")
IMG_SIZE     = 512
METAL_HU_DEF = 2500.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',      required=True,
                   help='Ścieżka do best_model.pth CircNetu')
    p.add_argument('--input-dir',       required=True,
                   help='Katalog z plikami metal*_slice*_H512_W512.raw')
    p.add_argument('--output-dir',      required=True,
                   help='Katalog wyjściowy dla *_circnet.raw')
    p.add_argument('--polar-size',      default='512x1024')
    p.add_argument('--radius-features', type=int, default=64)
    p.add_argument('--metal-hu',        type=float, default=METAL_HU_DEF)
    p.add_argument('--scan-ids',        type=int, nargs='+', default=None,
                   help='Opcjonalnie: ogranicz do wybranych scan ID')
    p.add_argument('--limit',           type=int, default=None,
                   help='Ogranicz liczbę plików (do testów)')
    p.add_argument('--device',          default='cuda')
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    n_polar = parse_polar_size(args.polar_size)

    # ── Załaduj CircNet ───────────────────────────────────────────────────────
    model = ScalarCorrectionNet(
        n_radii=n_polar[0],
        radius_features=args.radius_features,
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f'[CircNet] Model załadowany: {args.model_path}')
    print(f'[CircNet] Urządzenie: {device}  polar: {n_polar}')

    # ── Zbierz pliki wejściowe ────────────────────────────────────────────────
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in input_dir.glob('*.raw')
        if REAL_RE.match(f.name)
    )
    if args.scan_ids is not None:
        files = [
            f for f in files
            if int(REAL_RE.match(f.name).group('scan')) in args.scan_ids
        ]
    if args.limit:
        files = files[:args.limit]

    print(f'[CircNet] Plików do przetworzenia: {len(files)}')

    # ── Inferencja ────────────────────────────────────────────────────────────
    for art_path in tqdm(files, desc='CircNet inference'):
        art_orig = np.fromfile(str(art_path), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
        art      = np.clip(art_orig, HU_MIN, HU_MAX)
        art_norm = normalize(art)

        mask    = (art_orig > args.metal_hu).astype(np.float32)
        centers = get_all_metal_centers(mask)

        if len(centers) == 0:
            print(f'  [WARN] Brak metalu w {art_path.name}, kopiuję bez zmian.')
            out_name = art_path.name.replace('.raw', '_circnet.raw')
            art_orig.astype(np.float32).tofile(output_dir / out_name)
            continue

        corrected_norm = correct_image_with_mask(
            art_norm, mask, model, centers, device, n_polar
        )
        corrected_hu = denormalize(corrected_norm)

        # Przywróć oryginalne wartości HU w strefie metalu
        corrected_hu = corrected_hu * (1.0 - mask) + art_orig * mask

        out_name = art_path.name.replace('.raw', '_circnet.raw')
        corrected_hu.astype(np.float32).tofile(output_dir / out_name)

    print(f'[CircNet] Gotowe → {output_dir}')


if __name__ == '__main__':
    main()
