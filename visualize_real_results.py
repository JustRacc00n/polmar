"""
visualize_real_results.py — Wizualizacja wyników inferencji PolMAR na danych rzeczywistych

Generuje PNG z porównaniem dla każdej klatki:
  [Artefakt] | [CircNet] | [Refiner Base+CircNet] | [Refiner Base+CircNet+SEG]

Uruchomienie:
    python3 visualize_real_results.py \
        --art-dir       /app/data/real_raw \
        --circnet-dir   /app/output/real_circnet \
        --refiner-dir   /app/output/real_refiner_angle \
        --refiner-seg-dir /app/output/real_refiner_angle_graph \
        --out-dir       /app/output/real_vis \
        --ww 3000 --wl 0

Bez wariantu SEG (gdy nie był liczony):
    python3 visualize_real_results.py \
        --art-dir       /app/data/real_raw \
        --circnet-dir   /app/output/real_circnet \
        --refiner-dir   /app/output/real_refiner_angle \
        --out-dir       /app/output/real_vis
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

REAL_RE  = re.compile(r"^metal(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+\.raw$")
IMG_SIZE = 512


def index_dir(directory: Path) -> dict:
    """Zwraca {(scan, slice): Path} dla plików metal*_slice*.raw (dowolny sufiks)."""
    result = {}
    for p in sorted(directory.glob("metal*.raw")):
        base = p.name
        for suffix in ("_circnet.raw", "_refiner.raw", "_seg_channel.raw", ".raw"):
            if base.endswith(suffix):
                base = base[: -len(suffix)] + ".raw"
                break
        m = REAL_RE.match(base)
        if m:
            key = (int(m.group("scan")), int(m.group("slice")))
            result[key] = p
    return result


def load_raw(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)


def hu_to_display(arr: np.ndarray, ww: float, wl: float) -> np.ndarray:
    """Okno HU → [0, 1] float."""
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def save_comparison(
    out_path: Path,
    panels: list,          # list of (title, arr_float32_HU)
    ww: float,
    wl: float,
    scan_id: int,
    slice_id: int,
):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.4),
                              facecolor="#0d1b2a")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.04)

    for ax, (title, arr) in zip(axes, panels):
        img = hu_to_display(arr, ww, wl)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.axis("off")
        ax.set_facecolor("#0d1b2a")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"metal{scan_id:02d}  slice{slice_id:04d}    WW={int(ww)}  WL={int(wl)}",
        color="#7a9ab8", fontsize=9, y=0.97,
    )

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor="#0d1b2a")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--art-dir",         required=True,
                   help="Katalog z oryginalnymi plikami metal*_slice*.raw")
    p.add_argument("--circnet-dir",     required=True,
                   help="Katalog z wynikami CircNet (*_circnet.raw)")
    p.add_argument("--refiner-dir",     required=True,
                   help="Katalog z wynikami Refinera Base+CircNet (*_refiner.raw)")
    p.add_argument("--refiner-seg-dir", default=None,
                   help="Katalog z wynikami Refinera Base+CircNet+SEG (opcjonalny)")
    p.add_argument("--out-dir",         required=True,
                   help="Katalog wyjściowy dla PNG")
    p.add_argument("--ww",   type=float, default=3000.0, help="Window width [HU]")
    p.add_argument("--wl",   type=float, default=0.0,    help="Window level [HU]")
    p.add_argument("--scan-ids", type=int, nargs="+", default=None,
                   help="Ogranicz do wybranych scan ID")
    p.add_argument("--limit", type=int, default=None,
                   help="Ogranicz łączną liczbę klatek (do testów)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    art_map      = index_dir(Path(args.art_dir))
    circnet_map  = index_dir(Path(args.circnet_dir))
    refiner_map  = index_dir(Path(args.refiner_dir))
    seg_map      = index_dir(Path(args.refiner_seg_dir)) if args.refiner_seg_dir else {}

    keys = set(art_map) & set(circnet_map) & set(refiner_map)
    if args.scan_ids:
        keys = {k for k in keys if k[0] in args.scan_ids}
    keys = sorted(keys)

    if args.limit:
        keys = keys[: args.limit]

    missing = set(art_map) - set(circnet_map)
    if missing:
        print(f"[WARN] Brak CircNet dla {len(missing)} klatek")
    missing = set(art_map) - set(refiner_map)
    if missing:
        print(f"[WARN] Brak Refiner(angle) dla {len(missing)} klatek")
    if args.refiner_seg_dir:
        missing = set(art_map) - set(seg_map)
        if missing:
            print(f"[WARN] Brak Refiner(angle+SEG) dla {len(missing)} klatek")

    print(f"Klatek do wizualizacji: {len(keys)}")
    print(f"WW={args.ww}  WL={args.wl}")

    for (scan_id, slice_id) in tqdm(keys, desc="Wizualizacja"):
        art      = load_raw(art_map[(scan_id, slice_id)])
        circnet  = load_raw(circnet_map[(scan_id, slice_id)])
        refiner  = load_raw(refiner_map[(scan_id, slice_id)])

        panels = [
            ("Artefakt",          art),
            ("CircNet",           circnet),
            ("Refiner\nBase+CircNet", refiner),
        ]
        if (scan_id, slice_id) in seg_map:
            ref_seg = load_raw(seg_map[(scan_id, slice_id)])
            panels.append(("Refiner\nBase+CircNet+SEG", ref_seg))

        out_name = f"metal{scan_id:02d}_slice{slice_id:04d}.png"
        save_comparison(
            out_dir / out_name,
            panels,
            ww=args.ww,
            wl=args.wl,
            scan_id=scan_id,
            slice_id=slice_id,
        )

    print(f"Gotowe → {out_dir}")


if __name__ == "__main__":
    main()
