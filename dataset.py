"""
dataset.py — Ładowanie danych i transformacja do przestrzeni biegunowej

Struktura katalogów danych (dataset RPI):
    <data_root>/
        body1/
            Baseline/   *img*.raw   — obrazy z artefaktem
            Target/     *img*.raw   — obrazy czyste (ground truth)
            Mask/       *mask*.raw  — maski metalu
        body2/
            ...

Obrazy: float32, 512×512 pikseli, wartości w skali Hounsfielda [HU].
"""

import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------------------
# Stałe globalne
# -------------------------------------------------------------------------
HU_MIN    = -1000   # dolne obcięcie normalizacji
HU_MAX    =  3000   # górne obcięcie normalizacji
HU_BIO_LO = -1000   # dolna granica zakresu biologicznego (powietrze)
HU_BIO_HI =  1900   # górna granica zakresu biologicznego (kość kortykalna)

# Domyślny rozmiar siatki polarnej jako tuple (N_radii, N_angles)
# Zgodny z --polar-size 512x1024 z run.sh kolegi
# cv2.warpPolar: dsize=(width, height) = (N_radii, N_angles)
DEFAULT_POLAR_SIZE: Tuple[int, int] = (512, 1024)


# -------------------------------------------------------------------------
# Pomocnicze funkcje I/O
# -------------------------------------------------------------------------
def load_raw(path, shape: Tuple[int, int] = (512, 512),
             dtype=np.float32) -> np.ndarray:
    """Wczytuje binarny plik .raw i reshapuje do `shape`."""
    return np.fromfile(str(path), dtype=dtype).reshape(shape)


def normalize(img: np.ndarray, hu_min: int = HU_MIN,
              hu_max: int = HU_MAX) -> np.ndarray:
    img = np.nan_to_num(img, nan=hu_min, posinf=hu_max, neginf=hu_min)
    return np.clip((img - hu_min) / (hu_max - hu_min), 0.0, 1.0).astype(np.float32)


def denormalize(img_norm: np.ndarray, hu_min: int = HU_MIN,
                hu_max: int = HU_MAX) -> np.ndarray:
    return (img_norm * (hu_max - hu_min) + hu_min).astype(np.float32)


# -------------------------------------------------------------------------
# Transformacja biegunowa
# -------------------------------------------------------------------------
def polar_cv2(img: np.ndarray, cx: float, cy: float,
              n: Tuple[int, int] = DEFAULT_POLAR_SIZE,
              is_mask: bool = False) -> Tuple[np.ndarray, float]:
    """
    Transformuje obraz do przestrzeni biegunowej wycentrowanej w (cx, cy).

    n = (N_radii, N_angles) — konwencja cv2.warpPolar: dsize=(width, height).
    Wyjście: [N_angles, N_radii] — każdy wiersz to profil jednego kąta.
    """
    r_max = float(np.hypot(max(cx, img.shape[1] - cx), max(cy, img.shape[0] - cy)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    polar = cv2.warpPolar(
        img.astype(np.float32),
        dsize=n,
        center=(cx, cy),
        maxRadius=r_max,
        flags=cv2.WARP_POLAR_LINEAR | interp,
    )
    return polar, r_max   # shape: [N_angles, N_radii]


# -------------------------------------------------------------------------
# Wykrywanie centrów metalu
# -------------------------------------------------------------------------
def get_all_metal_centers(msk: np.ndarray,
                           min_area: int = 10) -> List[Tuple[float, float]]:
    """Zwraca listę centrów (cx, cy) dla każdego oddzielnego obiektu metalicznego."""
    msk_bin = (msk > 0).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        msk_bin, connectivity=8
    )
    centers = [
        tuple(centroids[i])
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] > min_area
    ]
    if not centers:
        centers.append((msk.shape[1] / 2.0, msk.shape[0] / 2.0))
    return centers


# -------------------------------------------------------------------------
# Budowanie listy próbek
# -------------------------------------------------------------------------
def find_subfolders(base_path: Path) -> Path:
    """Lokalizuje katalog z podfolderami Baseline/Target/Mask."""
    if (base_path / 'Target').exists():
        return base_path
    subdirs = [x for x in base_path.iterdir() if x.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else base_path


def idx_map(files, pattern=r'img(\d+)'):
    result = {}
    for f in files:
        match = re.search(pattern, f.name)
        if match:
            result[int(match.group(1))] = f
    return result


def build_paired_dataset(base_dirs: List[Path]) -> List[Tuple]:
    """
    Buduje listę próbek: (art_path, tgt_path, msk_path, cx, cy).
    Każdy obiekt metaliczny na obrazie tworzy osobną próbkę.

    base_dirs to lista ścieżek do folderów body* zawierających
    podfoldery Baseline/, Target/, Mask/.
    """
    all_samples = []
    for base in base_dirs:
        actual_path = find_subfolders(Path(base))
        baseline_dir = actual_path / 'Baseline'
        target_dir   = actual_path / 'Target'
        mask_dir     = actual_path / 'Mask'

        art_map_d  = idx_map(sorted(baseline_dir.glob('*img*.raw')))
        tgt_map_d  = idx_map(sorted(target_dir.glob('*img*.raw')))
        mask_map_d = idx_map(sorted(mask_dir.glob('*mask*.raw')))

        for i in sorted(set(art_map_d) & set(tgt_map_d) & set(mask_map_d)):
            msk = load_raw(mask_map_d[i])
            for (cx, cy) in get_all_metal_centers(msk):
                all_samples.append((art_map_d[i], tgt_map_d[i], mask_map_d[i], cx, cy))

    return all_samples


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class PolarScalarDataset(Dataset):
    """
    Dataset dla korekcji per piksel w przestrzeni polarnej.

    Dla każdej próbki (centrum metalu) oblicza mapę korekcji:

        c_target(θ, r) = clip(art_polar(θ,r) − tgt_polar(θ,r), 0) * non_metal(θ,r)

    Per-piksel cel (bez uśredniania wzdłuż r), zerowany na metalu.
    Tylko jasne artefakty (> 0) są do korekcji — ciemnych pasm nie naprawiamy.

    Zwraca 6 elementów:
        x_pol      : [2, N_a, N_r]  wejście sieci (artefakt + maska w biegunowej)
        c_target   : [N_a, N_r]     docelowa mapa korekcji per piksel
        angle_mask : [N_a]          1 = kąt przechodzi przez metal
        art_polar  : [N_a, N_r]     obraz artefaktu w biegunowej (do range loss)
        center     : [2]            (cx, cy) centrum metalu
        r_max      : skalar         promień transformacji
    """

    def __init__(self, samples: List[Tuple],
                 hu_min: int = HU_MIN,
                 hu_max: int = HU_MAX,
                 n_polar: Tuple[int, int] = DEFAULT_POLAR_SIZE):
        self.samples = samples
        self.hu_min  = hu_min
        self.hu_max  = hu_max
        self.n_polar = n_polar  # (N_radii, N_angles) — konwencja cv2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        art_path, tgt_path, msk_path, cx, cy = self.samples[idx]

        art = np.clip(load_raw(art_path), self.hu_min, self.hu_max)
        tgt = np.clip(load_raw(tgt_path), self.hu_min, self.hu_max)
        msk = load_raw(msk_path)

        # Transformacja polarna — oba obrazy do tej samej siatki
        art_polar, r_max = polar_cv2(art, cx, cy, n=self.n_polar)
        tgt_polar, _     = polar_cv2(tgt, cx, cy, n=self.n_polar)
        msk_polar, _     = polar_cv2(msk, cx, cy, n=self.n_polar, is_mask=True)

        art_n = normalize(art_polar, self.hu_min, self.hu_max)   # [N_a, N_r]
        tgt_n = normalize(tgt_polar, self.hu_min, self.hu_max)   # [N_a, N_r]
        msk_b = (msk_polar > 0.5).astype(np.float32)             # [N_a, N_r]

        # Cel per-piksel: różnica artefakt-czyste, zerowana na metalu.
        # Bez uśredniania wzdłuż r — model dostaje gęsty sygnał dla każdego piksela.
        # Tylko jasne artefakty (> 0): ciemnych pasm nie korygujemy (brak informacji).
        non_metal  = 1.0 - msk_b                                  # [N_a, N_r]
        c_target   = np.maximum(art_n - tgt_n, 0.0) * non_metal  # [N_a, N_r] >= 0

        # Maska kątów ze smugą artefaktu.
        # NAPRAWKA: poprzednio msk_b.max(axis=1) > 0 dawało ≈ 1 dla KAŻDEGO
        # kąta, bo centroid leży wewnątrz metalu → msk_b[:,0] ≈ 1 wszędzie
        # → zero_reg ≈ 0 (regularyzacja była martwa przez cały trening).
        # Teraz: c_target jest już wyzerowane na metalu, więc jego max > 0
        # wykrywa wyłącznie kąty z faktyczną smugą artefaktu poza metalem.
        # Próg 1e-3 ≈ 4 HU — odcina szum interpolacji polarnej.
        angle_mask = (c_target.max(axis=1) > 1e-3).astype(np.float32)  # [N_a]

        # Wejście sieci: artefakt + maska w przestrzeni biegunowej
        x_pol = np.stack([art_n, msk_b], axis=0)                 # [2, N_a, N_r]

        return (
            torch.from_numpy(x_pol).float(),             # [2, N_a, N_r]
            torch.from_numpy(c_target).float(),          # [N_a, N_r]
            torch.from_numpy(angle_mask).float(),        # [N_a]
            torch.from_numpy(art_n).float(),             # [N_a, N_r]
            torch.tensor([cx, cy]).float(),              # [2]
            torch.tensor(r_max).float(),                 # skalar
        )
