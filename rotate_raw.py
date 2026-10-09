"""
rotate_raw.py — Obraca wszystkie pliki .raw w podanych folderach o 90° w prawo (in-place)

Założenia:
  - float32, 512×512 pikseli
  - Obrót w prawo = np.rot90(arr, k=-1)  (k=3 równoważnie)

Uruchomienie:
    python3 rotate_raw.py
"""

import numpy as np
from pathlib import Path

# -------------------------------------------------------------------------
# Konfiguracja
# -------------------------------------------------------------------------
FOLDERS = [
    "/net/storage/pr3/plgrid/plggsmartudlteam/output/3frames_jitter_raw_clean_real",
    "/net/storage/pr3/plgrid/plggsmartudlteam/data/Clean_raw",
    "/net/storage/pr3/plgrid/plggsmartudlteam/data/real_raw",
]

IMG_SHAPE = (512, 512)
DTYPE     = np.float32

# -------------------------------------------------------------------------
# Obrót
# -------------------------------------------------------------------------
def rotate_right_90(arr: np.ndarray) -> np.ndarray:
    """Obrót o 90° w prawo (zgodnie z ruchem wskazówek zegara)."""
    return np.rot90(arr, k=-1)   # k=-1 = 90° CW


def process_folder(folder: str) -> None:
    p = Path(folder)
    if not p.exists():
        print(f"[WARN] Folder nie istnieje: {folder}")
        return

    raw_files = sorted(p.glob("*.raw"))
    if not raw_files:
        print(f"[INFO] Brak plików .raw w: {folder}")
        return

    print(f"\n[INFO] Folder: {folder}  ({len(raw_files)} plików)")

    ok = 0
    for i, fpath in enumerate(raw_files, 1):
        try:
            arr = np.fromfile(str(fpath), dtype=DTYPE)

            expected = IMG_SHAPE[0] * IMG_SHAPE[1]
            if arr.size != expected:
                print(f"  [{i:4d}/{len(raw_files)}] SKIP (rozmiar {arr.size} ≠ {expected}): {fpath.name}")
                continue

            arr = arr.reshape(IMG_SHAPE)
            arr = rotate_right_90(arr)
            arr.astype(DTYPE).tofile(str(fpath))   # nadpisz in-place

            ok += 1
            if i % 50 == 0 or i == len(raw_files):
                print(f"  [{i:4d}/{len(raw_files)}] OK")

        except Exception as e:
            print(f"  [{i:4d}/{len(raw_files)}] BŁĄD: {fpath.name} — {e}")

    print(f"[DONE] {ok}/{len(raw_files)} plików obróconych.")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Obrót 90° w prawo (in-place) — START")
    for folder in FOLDERS:
        process_folder(folder)
    print("\nWszystko gotowe.")
