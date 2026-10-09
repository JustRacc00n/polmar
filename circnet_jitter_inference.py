"""
circnet_jitter_inference.py — Inferencja CircNetu na datasecie szpitalnym (jitter)

Wczytuje pliki: jitter_Clean{NN}_slice{NNNN}_H512_W512.raw
Zapisuje pliki: jitter_Clean{NN}_slice{NNNN}_H512_W512_circnet.raw

Maska metalu derywowana z HU > --metal-hu (domyślnie 2500) — brak osobnych plików mask.

radius_features i polar_size są odczytywane automatycznie z best_model_args.json
(zapisywanego przez train.py obok best_model.pth).
Można nadpisać przez --radius-features / --polar-size.

Uruchomienie:
    python3 circnet_jitter_inference.py \
        --model-path  /app/CircNet/mar_run_16838558/best_model.pth \
        --input-dir   /app/output/3frames_jitter_raw_clean_real \
        --output-dir  /app/output/circnet_jitter_results
"""

import argparse
import json
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

# -------------------------------------------------------------------------
# Stałe
# -------------------------------------------------------------------------
ART_RE       = re.compile(r"^jitter_Clean(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+\.raw$")
IMG_SIZE     = 512
METAL_HU_DEF = 2500.0


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',      type=str, required=True,
                   help='Ścieżka do best_model.pth CircNetu')
    p.add_argument('--input-dir',       type=str, required=True,
                   help='Katalog z plikami jitter_Clean*_slice*_H512_W512.raw')
    p.add_argument('--output-dir',      type=str, required=True,
                   help='Katalog wyjściowy dla *_circnet.raw')
    p.add_argument('--polar-size',      type=str, default=None,
                   help='Nadpisuje polar_size z best_model_args.json, np. 512x1024')
    p.add_argument('--radius-features', type=int, default=None,
                   help='Nadpisuje radius_features z best_model_args.json')
    p.add_argument('--metal-hu',        type=float, default=METAL_HU_DEF,
                   help='Próg HU do derywacji maski metalu (domyślnie 2500)')
    p.add_argument('--limit',           type=int, default=None,
                   help='Ogranicz liczbę plików (do testów)')
    p.add_argument('--device',          type=str, default='cuda')
    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def _load_model_args(model_path: Path) -> dict:
    """Ładuje best_model_args.json z katalogu modelu (jeśli istnieje)."""
    args_path = model_path.parent / 'best_model_args.json'
    if args_path.exists():
        with open(args_path) as f:
            return json.load(f)
    return {}


def _infer_radius_features(state: dict) -> int:
    """Wyciąga radius_features z kształtu tensora w state dict jako fallback."""
    key = 'radius_proj.proj.3.weight'   # Linear(hidden → radius_features), idx 3 w Sequential
    if key in state:
        return state[key].shape[0]
    raise RuntimeError(
        f"Nie można odczytać radius_features: nie ma pliku best_model_args.json "
        f"ani klucza '{key}' w state dict. Podaj --radius-features ręcznie."
    )


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Odczytaj parametry modelu (JSON → fallback na state dict → fallback na CLI) ---
    model_path  = Path(args.model_path)
    saved_args  = _load_model_args(model_path)

    state = torch.load(model_path, map_location=device)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    if args.radius_features is not None:
        radius_features = args.radius_features          # CLI nadpisuje wszystko
    elif 'radius_features' in saved_args:
        radius_features = saved_args['radius_features'] # z best_model_args.json
    else:
        radius_features = _infer_radius_features(state) # fallback: kształt tensora

    polar_size_str = args.polar_size or saved_args.get('polar_size', '512x1024')
    n_polar = parse_polar_size(polar_size_str)

    # --- Załaduj model ---
    model = ScalarCorrectionNet(
        n_radii=n_polar[0],
        radius_features=radius_features,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f'Model załadowany z {model_path}')
    print(f'Polar size: {n_polar}  radius_features: {radius_features}')
    if saved_args:
        print(f'(parametry z best_model_args.json)')
    print(f'Metal HU próg: {args.metal_hu}')

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Zbierz pliki wejściowe ---
    all_files = sorted(
        p for p in input_dir.glob('jitter_Clean*_slice*_H512_W512.raw')
        if ART_RE.match(p.name)
    )
    if args.limit:
        all_files = all_files[:args.limit]
    print(f'Znaleziono plików: {len(all_files)}')

    # Resume: pomiń już istniejące
    tasks   = [(p, output_dir / f'{p.stem}_circnet.raw') for p in all_files]
    skipped = sum(1 for _, o in tasks if o.exists())
    tasks   = [(i, o) for i, o in tasks if not o.exists()]
    print(f'Do przetworzenia: {len(tasks)}  (pominięto gotowych: {skipped})')

    ok = err = no_metal = 0
    for art_path, out_path in tqdm(tasks, desc='CircNet inference (jitter)'):
        try:
            art_orig   = np.fromfile(str(art_path), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
            metal_mask = (art_orig > args.metal_hu).astype(np.float32)
            centers    = get_all_metal_centers(metal_mask)

            if not centers:
                # Brak metalu — zapisz obraz bez zmian
                art_orig.astype(np.float32).tofile(str(out_path))
                no_metal += 1
                continue

            art      = np.clip(art_orig, HU_MIN, HU_MAX)
            art_norm = normalize(art)

            corrected_norm = correct_image_with_mask(
                art_norm, metal_mask, model, centers, device, n_polar
            )
            corrected_hu = denormalize(corrected_norm)

            # Przywróć oryginalne HU w strefie metalu
            corrected_hu = corrected_hu * (1.0 - metal_mask) + art_orig * metal_mask
            corrected_hu.astype(np.float32).tofile(str(out_path))
            ok += 1

        except Exception as e:
            print(f'  [ERR] {art_path.name}: {e}')
            err += 1

    print(f'\nGotowe. ok={ok}  no_metal={no_metal}  err={err}  skip={skipped}')
    print(f'Wyniki zapisane w: {output_dir}')


if __name__ == '__main__':
    main()
