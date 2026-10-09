"""
real_refiner_inference.py — Refiner inference na rzeczywistych danych szpitalnych

Obsługuje dwa warianty:
  Base+CircNet         (2 kanały: art, circnet)
  Base+CircNet+SEG     (3 kanały: art, circnet, seg_channel)

Wzorzec pliku art:        metal{NN}_slice{NNNN}_H512_W512.raw
Wzorzec CircNet prior:    metal{NN}_slice{NNNN}_H512_W512_circnet.raw
Wzorzec SEG channel:      metal{NN}_slice{NNNN}_H512_W512_seg_channel.raw
Wzorzec wyjścia:          metal{NN}_slice{NNNN}_H512_W512_refiner.raw

Brak ground truth — skrypt zapisuje tylko predykcję, bez metryk.

Uruchomienie (Base+CircNet):
    python3 real_refiner_inference.py \
        --model-path   /app/CircNet/runs/refiner_jitter_angle_128f_18583599/best_refiner.pth \
        --art-dir      /app/data/real_raw \
        --circnet-dir  /app/output/real_circnet \
        --output-dir   /app/output/real_refiner_angle \
        --base-features 128

Uruchomienie (Base+CircNet+SEG):
    python3 real_refiner_inference.py \
        --model-path   /app/CircNet/runs/refiner_jitter_angle-graph_128f_18583601/best_refiner.pth \
        --art-dir      /app/data/real_raw \
        --circnet-dir  /app/output/real_circnet \
        --seg-chan-dir /app/data/real_seg_channels \
        --output-dir   /app/output/real_refiner_angle_graph \
        --base-features 128
"""

import argparse
import re
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# ── Stałe ─────────────────────────────────────────────────────────────────────
REAL_RE    = re.compile(r"^metal(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+\.raw$")
IMG_SIZE   = 512
HU_MIN     = -1000.0
HU_MAX     =  1000.0
METAL_HU   = 2500.0


def normalize(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)


def denormalize(x: np.ndarray) -> np.ndarray:
    return x * (HU_MAX - HU_MIN) + HU_MIN


# ── Model (identyczny jak w jitter_refiner_train.py) ──────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()
        b = base
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = ConvBlock(b,     b*2)
        self.enc3 = ConvBlock(b*2,   b*4)
        self.enc4 = ConvBlock(b*4,   b*8)
        self.pool = nn.MaxPool2d(2)
        self.up3  = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = ConvBlock(b*8, b*4)
        self.up2  = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = ConvBlock(b*4, b*2)
        self.up1  = nn.ConvTranspose2d(b*2, b,   2, stride=2)
        self.dec1 = ConvBlock(b*2, b)
        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def forward_refine(
    model:    nn.Module,
    x:        torch.Tensor,   # [B, C, H, W]
    mask:     torch.Tensor,   # [B, 1, H, W]
    art_orig: torch.Tensor,   # [B, 1, H, W]  oryginalne HU (znorm.)
) -> np.ndarray:
    """
    Zwraca predykcję [H, W] w jednostkach HU.
    Prior = kanał 1 (circnet).
    Hard paste: w strefie metalu = oryginał.
    """
    with torch.no_grad():
        delta = model(x)                                   # [B, 1, H, W]
        prior = x[:, 1:2]                                 # circnet channel
        pred  = prior + (1.0 - mask) * delta              # residual poza metalem
        pred  = pred * (1.0 - mask) + art_orig * mask     # hard paste metalu
    return denormalize(pred.squeeze().cpu().numpy())


# ── Indeksowanie plików ────────────────────────────────────────────────────────
def index_dir(directory: Path, suffix: str = '.raw') -> dict:
    """Zwraca {(scan, slice): Path} dla plików pasujących do REAL_RE."""
    result = {}
    for p in sorted(directory.glob(f'*{suffix}')):
        # Usuń znany sufiks przed dopasowaniem
        base = p.name
        for known in ('_circnet.raw', '_seg_channel.raw', '_refiner.raw', '.raw'):
            if base.endswith(known):
                base = base[:-len(known)] + '.raw'
                break
        m = REAL_RE.match(base)
        if m:
            key = (int(m.group('scan')), int(m.group('slice')))
            result[key] = p
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',    required=True,
                   help='Ścieżka do best_refiner.pth')
    p.add_argument('--art-dir',       required=True,
                   help='Katalog z oryginalnymi plikami metal*_slice*.raw')
    p.add_argument('--circnet-dir',   required=True,
                   help='Katalog z wynikami CircNetu (*_circnet.raw)')
    p.add_argument('--seg-chan-dir',  default=None,
                   help='Katalog z kanałami SEG (*_seg_channel.raw). '
                        'Jeśli podany → wariant 3-kanałowy Base+CircNet+SEG')
    p.add_argument('--output-dir',    required=True,
                   help='Katalog wyjściowy dla *_refiner.raw')
    p.add_argument('--base-features', type=int, default=128,
                   help='base_features UNetu (musi zgadzać się z treningiem)')
    p.add_argument('--metal-hu',      type=float, default=METAL_HU)
    p.add_argument('--scan-ids',      type=int, nargs='+', default=None,
                   help='Ogranicz do wybranych scan ID')
    p.add_argument('--device',        default='cuda')
    return p.parse_args()


def main():
    args      = parse_args()
    device    = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_seg   = args.seg_chan_dir is not None
    in_ch     = 3 if use_seg else 2
    variant   = 'Base+CircNet+SEG' if use_seg else 'Base+CircNet'

    # ── Załaduj model ─────────────────────────────────────────────────────────
    model = UNet(in_ch=in_ch, out_ch=1, base=args.base_features).to(device)
    state = torch.load(args.model_path, map_location=device)
    # Checkpointy zapisane jako pełny stan treningu: {'model': state_dict, 'epoch': ..., 'opt': ..., ...}
    if 'model' in state:
        state = state['model']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f'[Refiner] Wariant: {variant}  base={args.base_features}  device={device}')
    print(f'[Refiner] Model: {args.model_path}')

    # ── Zbierz pliki ──────────────────────────────────────────────────────────
    art_dir     = Path(args.art_dir)
    circnet_dir = Path(args.circnet_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    art_map     = index_dir(art_dir)
    circnet_map = index_dir(circnet_dir)
    seg_map     = index_dir(Path(args.seg_chan_dir)) if use_seg else {}

    # Klucze wspólne dla wszystkich wymaganych wejść
    keys = set(art_map) & set(circnet_map)
    if use_seg:
        keys &= set(seg_map)

    if args.scan_ids is not None:
        keys = {k for k in keys if k[0] in args.scan_ids}

    keys = sorted(keys)
    print(f'[Refiner] Próbek do przetworzenia: {len(keys)}')

    missing_circnet = set(art_map) - set(circnet_map)
    if missing_circnet:
        print(f'[WARN] Brak CircNet dla {len(missing_circnet)} plików — pomiń lub uruchom krok 1.')
    if use_seg:
        missing_seg = set(art_map) - set(seg_map)
        if missing_seg:
            print(f'[WARN] Brak SEG dla {len(missing_seg)} plików — pomiń lub uruchom krok 2.')

    # ── Inferencja ────────────────────────────────────────────────────────────
    for (scan_id, slice_id) in tqdm(keys, desc=f'Refiner ({variant})'):
        art_orig_hu = np.fromfile(str(art_map[(scan_id, slice_id)]),
                                  dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
        circnet_hu  = np.fromfile(str(circnet_map[(scan_id, slice_id)]),
                                  dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)

        art_n     = normalize(art_orig_hu)
        circnet_n = normalize(circnet_hu)
        mask      = (art_orig_hu > args.metal_hu).astype(np.float32)

        channels = [art_n, circnet_n]
        if use_seg:
            seg = np.fromfile(str(seg_map[(scan_id, slice_id)]),
                              dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
            channels.append(seg)

        x = torch.from_numpy(
            np.stack(channels, axis=0)
        ).unsqueeze(0).float().to(device)                          # [1, C, 512, 512]

        m_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
        a_t = torch.from_numpy(normalize(art_orig_hu)).unsqueeze(0).unsqueeze(0).float().to(device)

        pred_hu = forward_refine(model, x, m_t, a_t)

        # Nazwa wyjściowa na wzorzec oryginału
        src_name = art_map[(scan_id, slice_id)].name
        out_name = src_name.replace('.raw', '_refiner.raw')
        pred_hu.astype(np.float32).tofile(output_dir / out_name)

    print(f'[Refiner] Gotowe → {output_dir}')


if __name__ == '__main__':
    main()
