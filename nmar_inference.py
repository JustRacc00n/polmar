"""
nmar_inference.py — Inferencja NMAR na datasecie jitter_Clean*.raw

Wczytuje pliki: jitter_Clean{NN}_slice{NNNN}_H512_W512.raw
Zapisuje pliki: jitter_Clean{NN}_slice{NNNN}_H512_W512_nmar2.raw

Uruchomienie:
    python3 nmar_inference.py \
        --input-dir  /app/output/3frames_jitter_raw_clean_real \
        --output-dir /app/output/nmar_results \
        --num-workers 8

Uwagi:
  - NMAR jest CPU-only (leapctype, FBP)
  - Każdy worker tworzy własną instancję NMAR (leapctype nie jest thread-safe)
  - Pliki już przetworzone są pomijane (resume)
"""

import argparse
import re
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# Importujemy kod NMAR ze smartudl_copy
sys.path.insert(0, str(Path(__file__).parent.parent / "smartudl_copy"))
from src.models.submodules.nmar2 import NMAR

# -------------------------------------------------------------------------
# Stałe
# -------------------------------------------------------------------------
FILE_RE = re.compile(r"^jitter_Clean(\d+)_slice(\d+)_H(\d+)_W(\d+)\.raw$")
IMG_SIZE = 512

# -------------------------------------------------------------------------
# Worker (jeden proces = jedna instancja NMAR)
# -------------------------------------------------------------------------
def _worker_init():
    """Inicjalizuje globalną instancję NMAR w danym procesie."""
    global _nmar
    _nmar = NMAR(img_size=IMG_SIZE, suppress_output=True)


def _process_one(task):
    """
    task = (in_path, out_path)
    Zwraca (in_path.name, status_str)
    """
    in_path, out_path = task
    try:
        arr = np.fromfile(str(in_path), dtype=np.float32)
        if arr.size != IMG_SIZE * IMG_SIZE:
            return in_path.name, f"error: nieoczekiwany rozmiar {arr.size}"
        img = arr.reshape(IMG_SIZE, IMG_SIZE)

        result = _nmar.process_single_item(img)

        result.astype(np.float32).tofile(str(out_path))
        return in_path.name, "ok"
    except Exception as e:
        return in_path.name, f"error: {e}"


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   type=str, required=True,
                   help="Katalog z plikami jitter_Clean*_slice*_H512_W512.raw")
    p.add_argument("--output-dir",  type=str, required=True,
                   help="Katalog wyjściowy dla *_nmar2.raw")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Liczba równoległych procesów (każdy = osobna instancja NMAR)")
    p.add_argument("--limit",       type=int, default=None,
                   help="Ogranicz do N plików (do testów)")
    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Zbierz pliki wejściowe
    all_files = sorted(p for p in in_dir.glob("jitter_Clean*_slice*_H512_W512.raw")
                       if FILE_RE.match(p.name))

    if args.limit:
        all_files = all_files[:args.limit]

    print(f"Znaleziono plików wejściowych: {len(all_files)}")

    # Przygotuj zadania (pomijaj już istniejące)
    tasks = []
    skipped = 0
    for in_path in all_files:
        out_name = in_path.stem + "_nmar2.raw"
        out_path = out_dir / out_name
        if out_path.exists():
            skipped += 1
        else:
            tasks.append((in_path, out_path))

    print(f"Do przetworzenia: {len(tasks)}  (pominięto już gotowych: {skipped})")

    if not tasks:
        print("Wszystkie pliki już przetworzone.")
        return

    ok = err = 0
    with Pool(processes=args.num_workers, initializer=_worker_init) as pool:
        for name, status in tqdm(
            pool.imap_unordered(_process_one, tasks),
            total=len(tasks),
            desc="NMAR inferencja",
        ):
            if status == "ok":
                ok += 1
            else:
                err += 1
                print(f"  [ERR] {name}: {status}")

    print(f"\nGotowe. ok={ok}  err={err}  skip={skipped}")
    print(f"Wyniki zapisane w: {out_dir}")


if __name__ == "__main__":
    main()
