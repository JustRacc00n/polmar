"""
graph_channel_generator.py — Generuje semantyczny kanał struktury z segmentacji nnUNet

Dla każdej klatki CT (identyfikowanej przez img_id):
  1. Ładuje wolumen CT + segmentację (.nii.gz)
  2. Wyciąga slice Z=0 (jedyna warstwa)
  3. Buduje graf struktur (organy, kości, tkanka, smugi artefaktów)
  4. Dodaje głosy z sąsiednich klatek (±z_neighbors, ten sam folder body)
  5. Propaguje etykiety organów przez obszar smugi (BFS po krawędziach adjacent_flare)
  6. Zapisuje "uzupełniony" kanał semantyczny jako .raw float32 512x512

Kanał semantyczny [0..1]:
  0.0              — tło poza ciałem
  1/8 = 0.125      — liver (wątroba)
  2/8 = 0.250      — bladder (pęcherz)
  3/8 = 0.375      — lungs (płuca)
  4/8 = 0.500      — kidneys (nerki)
  5/8 = 0.625      — bone (kość)
  6/8 = 0.750      — metal
  7/8 = 0.875      — other_tissue (pozostała tkanka)
  smuga (flare)    — etykieta przewidywanej struktury (jak wyżej)
  nieznana smuga   — 0.875 (other_tissue — safe default)

Uruchomienie:
    python3 graph_channel_generator.py \
        --ct-dir      /app/data/CT_nii \
        --seg-dir     /app/data/SEG_nii \
        --out-dir     /app/data/seg_channels \
        --num-workers 8 \
        --z-neighbors 2 \
        --z-decay     0.7
"""

import argparse
import re
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.ndimage import (
    binary_closing, binary_fill_holes, binary_dilation,
    label as nd_label, distance_transform_edt,
)
from skimage.measure import regionprops, label as sk_label

# -------------------------------------------------------------------------
# Stałe
# -------------------------------------------------------------------------
CT_RE  = re.compile(r"body(?P<body>\d+)_img(?P<img>\d+)_0000\.nii\.gz$")
SEG_RE = re.compile(r"body(?P<body>\d+)_img(?P<img>\d+)_fullseg\.nii\.gz$")

# Tryb body_raw: CT z .raw, segmentacja z nnUNet .nii.gz
CT_RAW_RE  = re.compile(r"training_body_metalart_img(?P<img>\d+)_512x512x1\.raw$")
SEG_NII_RE = re.compile(r"body(?P<body>\d+)_img(?P<img>\d+)\.nii\.gz$")

# Tryb jitter_nii: oba pliki NII, nazwa jitter_Clean{scan}_slice{slice}
# CT NII:  jitter_Clean{NN}_slice{NNNN}_H512_W512_0000.nii.gz
# SEG NII: jitter_Clean{NN}_slice{NNNN}_H512_W512.nii.gz  (bez _0000)
JITTER_CT_RE  = re.compile(
    r"jitter_Clean(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+_0000\.nii\.gz$")
JITTER_SEG_RE = re.compile(
    r"jitter_Clean(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+\.nii\.gz$")

# Tryb real_nii: rzeczywiste dane szpitalne, nazwa metal{scan}_slice{slice}
# CT NII:  metal{NN}_slice{NNNN}_H512_W512_0000.nii.gz
# SEG NII: metal{NN}_slice{NNNN}_H512_W512.nii.gz  (bez _0000)
REAL_CT_RE  = re.compile(
    r"metal(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+_0000\.nii\.gz$")
REAL_SEG_RE = re.compile(
    r"metal(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+\.nii\.gz$")

# Klucz indeksu: (body_id, img_id) → Path
# Umożliwia wyszukiwanie sąsiednich klatek wyłącznie w obrębie tego samego body

LABELS = {
    "background": 0,
    "liver":      1,
    "bladder":    2,
    "lungs":      3,
    "kidneys":    4,
    "bone":       5,
    "metal":      6,
    "flare":      7,
}
LABEL_TO_VAL = {
    "background":   0.000,
    "liver":        1/8,
    "bladder":      2/8,
    "lungs":        3/8,
    "kidneys":      4/8,
    "bone":         5/8,
    "metal":        6/8,
    "other_tissue": 7/8,
    "unknown":      7/8,   # safe default: other_tissue
}
# Mapa ID etykiety → nazwa
ID_TO_NAME = {v: k for k, v in LABELS.items()}

# -------------------------------------------------------------------------
# Geometria
# -------------------------------------------------------------------------
def largest_cc(mask):
    lab, n = nd_label(mask.astype(np.uint8))
    if n == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def cc_regions(mask, min_area=1):
    lab = sk_label(mask.astype(np.uint8), connectivity=1)
    props = [r for r in regionprops(lab) if r.area >= min_area]
    props.sort(key=lambda r: (r.centroid[1], r.centroid[0], -r.area))
    return props


def mask_from_region(shape, region):
    m = np.zeros(shape, dtype=bool)
    m[tuple(zip(*region.coords))] = True
    return m


def contact_pixels(mask_a, mask_b, r=1):
    if not mask_a.any() or not mask_b.any():
        return 0
    dil = binary_dilation(mask_a, iterations=max(1, r))
    return int(np.logical_and(dil, mask_b).sum())


# -------------------------------------------------------------------------
# Podział smugi na fragmenty izodystansowe (pierścienie × sektory)
# -------------------------------------------------------------------------
def flare_parts_heightmap(flare_mask, L=5, K=6):
    parts = []
    H, W = flare_mask.shape
    for j, reg in enumerate(cc_regions(flare_mask, min_area=1)):
        parent = mask_from_region((H, W), reg)
        D = distance_transform_edt(parent)
        ys, xs = np.nonzero(parent)
        dvals = D[ys, xs]
        dmax = float(dvals.max())
        if dmax <= 0:
            continue

        # Pierścienie: quantile-based
        pos = dvals[dvals > 0]
        if pos.size >= max(2, L):
            qs = np.linspace(0, 1, L + 1)
            edges = np.quantile(pos, qs)
            edges[0] = 0.0
            edges[-1] = max(edges[-1], dmax + 1e-6)
        else:
            edges = np.linspace(0, dmax + 1e-6, L + 1)

        cy, cx = reg.centroid
        thetas = np.arctan2(ys - cy, xs - cx)

        # ring_idx 0 = zewnętrzny (mały D), L-1 = wewnętrzny (duży D)
        ring_idx   = np.digitize(D[ys, xs], edges[1:-1], right=False)   # 0..L-1
        sector_idx = np.clip(
            np.floor((thetas + np.pi) / (2 * np.pi) * K).astype(int), 0, K - 1
        )

        for l in range(L):
            in_ring = ring_idx == l
            if not in_ring.any():
                continue
            for k in range(K):
                sel = in_ring & (sector_idx == k)
                if not sel.any():
                    continue
                pmask = np.zeros_like(parent, dtype=bool)
                pmask[ys[sel], xs[sel]] = True
                for isl_idx, ir in enumerate(cc_regions(pmask, min_area=1)):
                    im = mask_from_region((H, W), ir)
                    parts.append({
                        "mask": im,
                        "flare_j": j,
                        "ring_l": l,
                        "sector_k": k,
                        "island_idx": isl_idx,
                        "depth_mean": float(D[im].mean()) if im.any() else 0.0,
                    })
    return parts


# -------------------------------------------------------------------------
# Graf — węzły + krawędzie
# -------------------------------------------------------------------------
def build_graph(ct_slice, seg_slice,
                T_body=-400.0, r_close=5,
                min_area_organ=200, min_area_other=500,
                max_cc_bone=60, flare_L=5, flare_K=6,
                r_adj_ff=1, r_adj_ft=1):
    """
    Zwraca:
      node_masks : dict {node_id: bool mask}
      tissue_ids : list[str]  — węzły tkankowe (organy + other)
      flare_ids  : list[str]  — węzły fragmentów smugi
      node_label : dict {node_id: label_name}
      ff_neighbors : dict {flare_id: [(neighbor_id, contact_ratio)]}
      ft_edges   : list[(flare_id, tissue_id, contact_ratio_src)]
    """
    H, W = ct_slice.shape

    # --- Body mask ---
    body = ct_slice > T_body
    body = binary_closing(body, iterations=max(1, r_close))
    body = binary_fill_holes(body)
    body = largest_cc(body)

    M = {name: (seg_slice == lab) for name, lab in LABELS.items()}
    M_org = M["liver"] | M["bladder"] | M["lungs"] | M["kidneys"] | M["bone"]
    M_other = body & ~(M_org | M["metal"] | M["flare"])

    node_masks  = {}
    node_label  = {}

    def add(nid, label, mask):
        if mask.any():
            node_masks[nid] = mask.copy()
            node_label[nid] = label

    # Organy
    for name in ["liver", "bladder"]:
        add(f"organ:{name}", name, M[name])

    for name, max_cc in [("lungs", 20), ("kidneys", 10), ("bone", max_cc_bone)]:
        for i, r in enumerate(cc_regions(M[name], min_area=min_area_organ)):
            if max_cc and i >= max_cc:
                break
            add(f"organ:{name}:{i}", name, mask_from_region((H, W), r))

    # Other tissue
    rest_mask = np.zeros((H, W), dtype=bool)
    for r in cc_regions(M_other, min_area=min_area_other):
        rest_mask |= mask_from_region((H, W), r)
    add("tissue:other", "other_tissue", rest_mask)

    tissue_ids = list(node_masks.keys())

    # Fragmenty smugi
    parts = flare_parts_heightmap(M["flare"], L=flare_L, K=flare_K)
    parts.sort(key=lambda p: (p["flare_j"], p["ring_l"], p["sector_k"], p["island_idx"]))
    flare_ids = []
    for idx, p in enumerate(parts):
        nid = f"flare:{idx}"
        add(nid, "flare", p["mask"])
        flare_ids.append(nid)

    # Krawędzie flare↔flare
    ff_neighbors = {fid: [] for fid in flare_ids}
    for i in range(len(flare_ids)):
        a = flare_ids[i]
        ma = node_masks.get(a)
        if ma is None or not ma.any():
            continue
        area_a = int(ma.sum())
        for j in range(i + 1, len(flare_ids)):
            b = flare_ids[j]
            mb = node_masks.get(b)
            if mb is None or not mb.any():
                continue
            area_b = int(mb.sum())
            cp = contact_pixels(ma, mb, r=r_adj_ff)
            if cp > 0:
                ff_neighbors[a].append((b, float(cp / area_a)))
                ff_neighbors[b].append((a, float(cp / area_b)))

    # Krawędzie flare→tissue
    ft_edges = []
    for fid in flare_ids:
        mf = node_masks.get(fid)
        if mf is None or not mf.any():
            continue
        area_f = int(mf.sum())
        for tid in tissue_ids:
            mt = node_masks.get(tid)
            if mt is None or not mt.any():
                continue
            cp = contact_pixels(mf, mt, r=r_adj_ft)
            if cp > 0:
                ft_edges.append((fid, tid, float(cp / area_f)))

    return node_masks, tissue_ids, flare_ids, node_label, ff_neighbors, ft_edges


# -------------------------------------------------------------------------
# Głosy z sąsiednich klatek Z
# -------------------------------------------------------------------------
def add_cross_slice_votes(
    scores,          # dict {fid: {label: float}} — modyfikowane w miejscu
    flare_ids,
    node_masks,
    neighbor_segs,   # list of (seg_slice np.ndarray int16, z_weight float)
):
    """
    Dla każdego fragmentu smugi sprawdza, jakie etykiety tkankowe istnieją
    w tych samych pikselach w sąsiednich klatkach i dodaje ważone głosy do scores.

    Głosy z sąsiednich klatek trafiają do scores PRZED BFS — dzięki temu
    uczestniczą też w propagacji do fragmentów bez żadnych sąsiadów 2D.

    Waga = z_decay^|dz| * fraction_of_pixels_with_that_label.
    Ignorowane: background (0), metal (6), flare (7).
    """
    IGNORE = {LABELS["background"], LABELS["metal"], LABELS["flare"]}

    for fid in flare_ids:
        mask = node_masks.get(fid)
        if mask is None or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        n_pix = float(len(ys))

        for nseg, z_w in neighbor_segs:
            neighbor_lab_ids = nseg[ys, xs]
            for lab_id in np.unique(neighbor_lab_ids):
                if int(lab_id) in IGNORE:
                    continue
                lab_name = ID_TO_NAME.get(int(lab_id), "other_tissue")
                if lab_name in ("background", "flare", "unknown"):
                    continue
                frac = float((neighbor_lab_ids == lab_id).sum()) / (n_pix + 1e-8)
                scores[fid][lab_name] = scores[fid].get(lab_name, 0.0) + frac * z_w


# -------------------------------------------------------------------------
# BFS propagacja etykiet do wnętrza smugi (POPRAWKA)
# -------------------------------------------------------------------------
def propagate_labels_bfs(
    flare_ids, tissue_ids, node_label,
    ff_neighbors, ft_edges,
    node_masks=None,
    neighbor_segs=None,
    use_contact_ratio=True,
    hop_decay=0.5,
    max_hops=10,
):
    """
    Przypisuje etykietę struktury każdemu fragmentowi smugi.

    Krok 1: bezpośrednie głosy od sąsiadów tkankowych (krawędzie flare→tissue)
            + opcjonalne głosy z sąsiednich klatek Z (neighbor_segs).
    Krok 2: BFS po krawędziach flare↔flare dla fragmentów bez sąsiedztwa tkankowego.

    POPRAWKA buga (propagacja BFS):
    - W kolejce BFS przekazywane są aktualnie zdyskontowane wyniki (decayed_scores),
      NIE oryginalne wyniki źródła (source_scores).
    - Waga zdyskontowania per hop: edge_w * hop_decay (mnożona w każdym kroku).
    - Dzięki temu ścieżka A→B→C akumuluje: score_A * w_AB * w_BC * hop_decay^2
      (zamiast błędnego: score_A * w_BC * hop_decay^2).

    neighbor_segs : list of (seg_slice np.ndarray, z_weight) lub None
    node_masks    : wymagane gdy neighbor_segs jest podane
    """
    # ---- Krok 1a: bezpośrednie głosy tkankowe (2D krawędzie flare→tissue) ----
    scores: dict[str, dict[str, float]] = {fid: {} for fid in flare_ids}

    for fid, tid, cr in ft_edges:
        lab = node_label.get(tid, "unknown")
        if lab in ("unknown", "flare", "background"):
            continue
        w = cr if use_contact_ratio else 1.0
        scores[fid][lab] = scores[fid].get(lab, 0.0) + w

    # ---- Krok 1b: głosy z sąsiednich klatek Z ----
    if neighbor_segs and node_masks is not None:
        add_cross_slice_votes(scores, flare_ids, node_masks, neighbor_segs)

    def best(score_dict):
        if not score_dict:
            return "unknown", 0.0
        labs = list(score_dict.keys())
        vals = np.array([score_dict[l] for l in labs], dtype=np.float32)
        vals /= (vals.sum() + 1e-8)
        k = int(np.argmax(vals))
        return labs[k], float(vals[k])

    results = {}
    for fid in flare_ids:
        lab, prob = best(scores[fid])
        results[fid] = {"label": lab, "prob": prob, "hops": 0 if lab != "unknown" else -1}

    # ---- Krok 2: BFS po krawędziach flare↔flare ----
    unknown_ids = {fid for fid in flare_ids if results[fid]["label"] == "unknown"}
    if not unknown_ids:
        return results

    propagated: dict[str, dict[str, float]] = {fid: {} for fid in unknown_ids}
    visited_hops: dict[str, int] = {}

    # Inicjalizacja kolejki — wszystkie już znane węzły z ich score
    queue = deque()
    for fid in flare_ids:
        if results[fid]["label"] != "unknown" and scores[fid]:
            queue.append((fid, scores[fid].copy(), 0))

    while queue:
        current_fid, current_scores, hop = queue.popleft()
        if hop >= max_hops:
            continue

        for neighbor_fid, edge_w in ff_neighbors.get(current_fid, []):
            if neighbor_fid not in unknown_ids:
                continue

            prev_hop = visited_hops.get(neighbor_fid, max_hops + 1)
            if hop + 1 > prev_hop:
                continue  # Doszliśmy tu krótszą ścieżką poprzednio — pomijamy

            # POPRAWKA: dyskontuj aktualne wyniki (nie oryginalne źródło)
            # Każdy krok mnoży sygnał przez edge_w * hop_decay
            decayed = {lab: w * edge_w * hop_decay
                       for lab, w in current_scores.items()}

            for lab, w in decayed.items():
                propagated[neighbor_fid][lab] = propagated[neighbor_fid].get(lab, 0.0) + w

            visited_hops[neighbor_fid] = hop + 1
            # POPRAWKA: do kolejki trafiają zdyskontowane wyniki, nie oryginalne
            queue.append((neighbor_fid, decayed, hop + 1))

    # Zaktualizuj wyniki
    for fid in unknown_ids:
        lab, prob = best(propagated[fid])
        if lab != "unknown":
            results[fid] = {"label": lab, "prob": prob,
                            "hops": visited_hops.get(fid, -1)}

    return results


# -------------------------------------------------------------------------
# Generowanie kanału semantycznego (512×512 float32)
# -------------------------------------------------------------------------
def generate_semantic_channel(ct_slice, seg_slice,
                               neighbor_segs=None,
                               T_body=-400.0, r_close=5,
                               min_area_organ=200, min_area_other=500,
                               flare_L=5, flare_K=6,
                               hop_decay=0.5, max_hops=10):
    """
    Zwraca float32 array [H, W] z wartościami z LABEL_TO_VAL.
    Piksele smugi (flare) dostają wartość przewidywanej struktury.

    neighbor_segs : list of (seg_slice np.ndarray int16, z_weight float) lub None.
        Etykiety z sąsiednich klatek są dodawane jako głosy przed BFS.
        Pozwala rozwiązać fragmenty smugi bez żadnych sąsiadów 2D.
    """
    H, W = ct_slice.shape
    channel = np.zeros((H, W), dtype=np.float32)

    # --- Organy i tkanka z segmentacji ---
    for name, lab_id in LABELS.items():
        if name in ("background", "flare"):
            continue
        val = LABEL_TO_VAL.get(name, 0.0)
        channel[seg_slice == lab_id] = val

    # Other tissue (body \ segmentowane klasy)
    body = ct_slice > T_body
    body = binary_closing(body, iterations=max(1, r_close))
    body = binary_fill_holes(body)
    body = largest_cc(body)
    M_known = (seg_slice > 0)
    M_other  = body & ~M_known
    rest = np.zeros((H, W), dtype=bool)
    for r in cc_regions(M_other, min_area=min_area_other):
        rest |= mask_from_region((H, W), r)
    channel[rest] = LABEL_TO_VAL["other_tissue"]

    # --- Graf + propagacja dla pikseli smugi ---
    if (seg_slice == LABELS["flare"]).any():
        try:
            node_masks, tissue_ids, flare_ids, node_label, ff_neighbors, ft_edges = \
                build_graph(ct_slice, seg_slice,
                            T_body=T_body, r_close=r_close,
                            min_area_organ=min_area_organ, min_area_other=min_area_other,
                            flare_L=flare_L, flare_K=flare_K)

            pred = propagate_labels_bfs(
                flare_ids, tissue_ids, node_label,
                ff_neighbors, ft_edges,
                node_masks=node_masks,
                neighbor_segs=neighbor_segs,
                hop_decay=hop_decay, max_hops=max_hops,
            )

            for fid in flare_ids:
                mask = node_masks.get(fid)
                if mask is None or not mask.any():
                    continue
                lab = pred.get(fid, {}).get("label", "unknown")
                val = LABEL_TO_VAL.get(lab, LABEL_TO_VAL["unknown"])
                channel[mask] = val

        except Exception as e:
            # Fallback: smuga dostaje wartość other_tissue
            flare_mask = (seg_slice == LABELS["flare"])
            channel[flare_mask] = LABEL_TO_VAL["unknown"]
            print(f"  [WARN] Graf nie powiódł się: {e}")

    return channel


# -------------------------------------------------------------------------
# Indeksowanie plików
# -------------------------------------------------------------------------
def index_files(ct_dir, seg_dir):
    """
    Zwraca:
      ct_map  : {img_id: (body_id, Path)}
      seg_map : {img_id: (body_id, Path)}
      common  : sorted list of img_ids obecnych w obu mapach
    """
    ct_map, seg_map = {}, {}
    for p in Path(ct_dir).glob("body*_img*_0000.nii.gz"):
        m = CT_RE.search(p.name)
        if m:
            ct_map[int(m.group("img"))] = (int(m.group("body")), p)
    for p in Path(seg_dir).glob("body*_img*_fullseg.nii.gz"):
        m = SEG_RE.search(p.name)
        if m:
            seg_map[int(m.group("img"))] = (int(m.group("body")), p)
    common = sorted(set(ct_map) & set(seg_map))
    return ct_map, seg_map, common


def get_neighbor_seg_specs(img_id, body_id, seg_map, n_neighbors, z_decay):
    """
    Zwraca listę (seg_path, z_weight) dla sąsiednich klatek tego samego body.
    Przeszukuje dz = 1..n_neighbors w obu kierunkach.
    """
    specs = []
    for dz in range(1, n_neighbors + 1):
        for sign in (-1, 1):
            nid = img_id + sign * dz
            if nid not in seg_map:
                continue
            nbody, npath = seg_map[nid]
            if nbody != body_id:
                continue   # inna sekwencja — pomijamy
            specs.append((npath, float(z_decay ** dz)))
    return specs


# -------------------------------------------------------------------------
# Worker (dla multiprocessing)
# -------------------------------------------------------------------------
def process_one(args):
    img_id, body_id, ct_path, seg_path, neighbor_seg_specs, out_dir, cfg = args
    out_path = Path(out_dir) / f"seg_channel_img{img_id}_512x512x1.raw"
    if out_path.exists():
        return img_id, "skip"

    try:
        ct  = nib.load(str(ct_path)).get_fdata().astype(np.float32)
        seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)

        ct_sl  = ct[:, :, 0]
        seg_sl = seg[:, :, 0]

        # Załaduj sąsiednie klatki (tylko SEG, CT sąsiada nie jest potrzebne)
        neighbor_segs = []
        for npath, z_w in neighbor_seg_specs:
            try:
                nseg = nib.load(str(npath)).get_fdata().astype(np.int16)[:, :, 0]
                neighbor_segs.append((nseg, z_w))
            except Exception:
                pass   # sąsiad niedostępny — pomijamy

        channel = generate_semantic_channel(
            ct_sl, seg_sl,
            neighbor_segs=neighbor_segs if neighbor_segs else None,
            T_body=cfg["T_body"], r_close=cfg["r_close"],
            min_area_organ=cfg["min_area_organ"],
            min_area_other=cfg["min_area_other"],
            flare_L=cfg["flare_L"], flare_K=cfg["flare_K"],
            hop_decay=cfg["hop_decay"], max_hops=cfg["max_hops"],
        )
        channel.astype(np.float32).tofile(str(out_path))
        return img_id, "ok"
    except Exception as e:
        return img_id, f"error: {e}"


# -------------------------------------------------------------------------
# Tryb body_raw — indeksowanie i worker
# -------------------------------------------------------------------------
def _find_body_root(folder_path: Path) -> Path:
    """
    Odpowiednik find_subfolders z dataset.py.
    Jeśli folder_path/Target istnieje — zwraca folder_path.
    Jeśli jest jeden podfolder — zwraca go (obsługuje zagnieżdżenie).
    """
    if (folder_path / 'Target').exists():
        return folder_path
    subdirs = [x for x in folder_path.iterdir() if x.is_dir()]
    return subdirs[0] if len(subdirs) == 1 else folder_path


def index_files_body_raw(data_root, seg_dir, body_folders):
    """
    Tryb body_raw:
      CT  — body{N}/Baseline/*img*.raw  (ta sama struktura co body1-5)
      SEG — body{N}_img{M}.nii.gz  (wyjście nnUNet)

    Zwraca:
      ct_map  : {img_id: (body_id, Path)}
      seg_map : {img_id: (body_id, Path)}
      common  : sorted list of img_ids obecnych w obu mapach
    """
    IMG_PAT = re.compile(r'img(\d+)')
    ct_map, seg_map = {}, {}

    # Zbiór body_id z body_folders do filtrowania segmentacji
    allowed_body_ids = set()
    for folder in body_folders:
        m = re.search(r"body(\d+)", folder)
        if m:
            allowed_body_ids.add(int(m.group(1)))

    # CT — szukaj w Baseline/ (jak dataset.py)
    for folder in body_folders:
        m_body = re.search(r"body(\d+)", folder)
        if not m_body:
            continue
        body_id     = int(m_body.group(1))
        folder_path = Path(data_root) / folder
        if not folder_path.exists():
            print(f"  [WARN] Folder nie istnieje: {folder_path}")
            continue
        actual_path  = _find_body_root(folder_path)
        baseline_dir = actual_path / 'Baseline'
        if not baseline_dir.exists():
            print(f"  [WARN] Brak katalogu Baseline: {baseline_dir}")
            continue
        for p in sorted(baseline_dir.glob('*img*.raw')):
            m = IMG_PAT.search(p.name)
            if m:
                ct_map[int(m.group(1))] = (body_id, p)
        print(f"  body{body_id}: znaleziono {sum(1 for bid, _ in ct_map.values() if bid == body_id)} plików CT")

    # SEG — body{N}_img{M}.nii.gz
    for p in Path(seg_dir).glob("body*_img*.nii.gz"):
        m = SEG_NII_RE.search(p.name)
        if m:
            body_id = int(m.group("body"))
            img_id  = int(m.group("img"))
            if body_id in allowed_body_ids:
                seg_map[img_id] = (body_id, p)
    print(f"  Segmentacji znaleziono: {len(seg_map)}")

    common = sorted(set(ct_map) & set(seg_map))
    return ct_map, seg_map, common


def _load_seg_nii(path) -> np.ndarray:
    """Ładuje segmentację z NII, zwraca int16 array [H, W]."""
    vol = nib.load(str(path)).get_fdata().astype(np.int16)
    # nnUNet może zwrócić (H, W, 1) lub (1, H, W) lub (H, W)
    if vol.ndim == 3:
        # wybierz wymiar o długości 1
        if vol.shape[2] == 1:
            return vol[:, :, 0]
        elif vol.shape[0] == 1:
            return vol[0]
        else:
            return vol[:, :, 0]   # fallback: pierwsza warstwa Z
    return vol


def process_one_body_raw(args):
    """
    Worker dla trybu body_raw.
    task = (img_id, body_id, ct_path, seg_path, neighbor_seg_specs, out_dir, cfg)
    CT ładowane z .raw float32 512×512, SEG z .nii.gz.
    """
    img_id, body_id, ct_path, seg_path, neighbor_seg_specs, out_dir, cfg = args
    out_path = Path(out_dir) / f"seg_channel_img{img_id}_512x512x1.raw"
    if out_path.exists():
        return img_id, "skip"

    try:
        # CT z pliku .raw (float32 512×512)
        ct_sl = np.fromfile(str(ct_path), dtype=np.float32).reshape(512, 512)

        # Segmentacja z NII
        seg_sl = _load_seg_nii(seg_path)

        # Sąsiednie klatki (tylko SEG)
        neighbor_segs = []
        for npath, z_w in neighbor_seg_specs:
            try:
                nseg = _load_seg_nii(npath)
                neighbor_segs.append((nseg, z_w))
            except Exception:
                pass

        channel = generate_semantic_channel(
            ct_sl, seg_sl,
            neighbor_segs=neighbor_segs if neighbor_segs else None,
            T_body=cfg["T_body"], r_close=cfg["r_close"],
            min_area_organ=cfg["min_area_organ"],
            min_area_other=cfg["min_area_other"],
            flare_L=cfg["flare_L"], flare_K=cfg["flare_K"],
            hop_decay=cfg["hop_decay"], max_hops=cfg["max_hops"],
        )
        channel.astype(np.float32).tofile(str(out_path))
        return img_id, "ok"
    except Exception as e:
        return img_id, f"error: {e}"


# -------------------------------------------------------------------------
# Tryb jitter_nii — indeksowanie i worker
# -------------------------------------------------------------------------
def index_files_jitter_nii(ct_dir, seg_dir):
    """
    Tryb jitter_nii:
      CT  — jitter_Clean{NN}_slice{NNNN}_H512_W512_0000.nii.gz
      SEG — jitter_Clean{NN}_slice{NNNN}_H512_W512.nii.gz  (bez _0000)

    img_id = scan_id * 100000 + slice_id  (unikalny klucz, umożliwia wyszukiwanie sąsiadów ±dz)
    body_id = scan_id  (używany do filtrowania sąsiadów z tego samego scanu)

    Zwraca:
      ct_map  : {img_id: (scan_id, Path)}
      seg_map : {img_id: (scan_id, Path)}
      common  : sorted list of img_ids obecnych w obu mapach
    """
    ct_map, seg_map = {}, {}

    for p in sorted(Path(ct_dir).glob("jitter_Clean*_0000.nii.gz")):
        m = JITTER_CT_RE.search(p.name)
        if m:
            scan_id  = int(m.group("scan"))
            slice_id = int(m.group("slice"))
            img_id   = scan_id * 100000 + slice_id
            ct_map[img_id] = (scan_id, p)

    for p in sorted(Path(seg_dir).glob("jitter_Clean*.nii.gz")):
        # Pomiń pliki z _0000 (to CT, nie SEG)
        if "_0000.nii.gz" in p.name:
            continue
        m = JITTER_SEG_RE.search(p.name)
        if m:
            scan_id  = int(m.group("scan"))
            slice_id = int(m.group("slice"))
            img_id   = scan_id * 100000 + slice_id
            seg_map[img_id] = (scan_id, p)

    # Diagnostyka
    scans_ct  = sorted({scan for scan, _ in ct_map.values()})
    scans_seg = sorted({scan for scan, _ in seg_map.values()})
    print(f"  CT:  {len(ct_map)} plików, scany: {scans_ct}")
    print(f"  SEG: {len(seg_map)} plików, scany: {scans_seg}")

    common = sorted(set(ct_map) & set(seg_map))
    return ct_map, seg_map, common


def _load_ct_nii(path) -> np.ndarray:
    """Ładuje CT z NII, zwraca float32 array [H, W]."""
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    if vol.ndim == 3:
        if vol.shape[2] == 1:
            return vol[:, :, 0]
        elif vol.shape[0] == 1:
            return vol[0]
        else:
            return vol[:, :, 0]   # fallback: pierwsza warstwa Z
    return vol


def index_files_real_nii(ct_dir, seg_dir):
    """
    Tryb real_nii:
      CT  — metal{NN}_slice{NNNN}_H512_W512_0000.nii.gz
      SEG — metal{NN}_slice{NNNN}_H512_W512.nii.gz  (bez _0000)

    img_id = scan_id * 100000 + slice_id  (unikalny klucz, umożliwia wyszukiwanie sąsiadów ±dz)
    body_id = scan_id  (używany do filtrowania sąsiadów z tego samego scanu)

    Zwraca:
      ct_map  : {img_id: (scan_id, Path)}
      seg_map : {img_id: (scan_id, Path)}
      common  : sorted list of img_ids obecnych w obu mapach
    """
    ct_map, seg_map = {}, {}

    for p in sorted(Path(ct_dir).glob("metal*_0000.nii.gz")):
        m = REAL_CT_RE.search(p.name)
        if m:
            scan_id  = int(m.group("scan"))
            slice_id = int(m.group("slice"))
            img_id   = scan_id * 100000 + slice_id
            ct_map[img_id] = (scan_id, p)

    for p in sorted(Path(seg_dir).glob("metal*.nii.gz")):
        # Pomiń pliki z _0000 (to CT, nie SEG)
        if "_0000.nii.gz" in p.name:
            continue
        m = REAL_SEG_RE.search(p.name)
        if m:
            scan_id  = int(m.group("scan"))
            slice_id = int(m.group("slice"))
            img_id   = scan_id * 100000 + slice_id
            seg_map[img_id] = (scan_id, p)

    # Diagnostyka
    scans_ct  = sorted({scan for scan, _ in ct_map.values()})
    scans_seg = sorted({scan for scan, _ in seg_map.values()})
    print(f"  CT:  {len(ct_map)} plików, scany: {scans_ct}")
    print(f"  SEG: {len(seg_map)} plików, scany: {scans_seg}")

    common = sorted(set(ct_map) & set(seg_map))
    return ct_map, seg_map, common


# process_one_real_nii reużywa process_one_jitter_nii:
# nazwa wyjściowa jest budowana tak samo (strip _0000.nii.gz → + _seg_channel.raw),
# co działa poprawnie zarówno dla metal{NN}_slice{NNNN} jak i jitter_Clean{NN}_slice{NNNN}.


def process_one_jitter_nii(args):
    """
    Worker dla trybów jitter_nii i real_nii.
    task = (img_id, scan_id, ct_path, seg_path, neighbor_seg_specs, out_dir, cfg)

    CT i SEG ładowane z .nii.gz.
    Plik wyjściowy: {art_stem}_seg_channel.raw
      gdzie art_stem = nazwa CT bez '_0000.nii.gz'
    """
    img_id, scan_id, ct_path, seg_path, neighbor_seg_specs, out_dir, cfg = args

    # Nazwa wyjściowa: usuń '_0000.nii.gz', dodaj '_seg_channel.raw'
    ct_name   = Path(ct_path).name                  # …_0000.nii.gz
    art_stem  = ct_name.replace("_0000.nii.gz", "") # jitter_Clean01_slice0000_H512_W512
    out_path  = Path(out_dir) / f"{art_stem}_seg_channel.raw"

    if out_path.exists():
        return img_id, "skip"

    try:
        ct_sl  = _load_ct_nii(ct_path)
        seg_sl = _load_seg_nii(seg_path)

        # Sąsiednie klatki (tylko SEG)
        neighbor_segs = []
        for npath, z_w in neighbor_seg_specs:
            try:
                nseg = _load_seg_nii(npath)
                neighbor_segs.append((nseg, z_w))
            except Exception:
                pass

        channel = generate_semantic_channel(
            ct_sl, seg_sl,
            neighbor_segs=neighbor_segs if neighbor_segs else None,
            T_body=cfg["T_body"], r_close=cfg["r_close"],
            min_area_organ=cfg["min_area_organ"],
            min_area_other=cfg["min_area_other"],
            flare_L=cfg["flare_L"], flare_K=cfg["flare_K"],
            hop_decay=cfg["hop_decay"], max_hops=cfg["max_hops"],
        )
        channel.astype(np.float32).tofile(str(out_path))
        return img_id, "ok"
    except Exception as e:
        return img_id, f"error: {e}"


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",           type=str, default="nii",
                   choices=["nii", "body_raw", "jitter_nii", "real_nii"],
                   help="nii: CT+SEG z NII (body dataset, fullseg); "
                        "body_raw: CT z .raw + SEG z nnUNet NII (dataset body); "
                        "jitter_nii: CT i SEG z NII — dataset jitter szpitalny; "
                        "real_nii: CT i SEG z NII — rzeczywiste dane szpitalne metal{NN}_slice{NNNN}")
    # -- tryb nii --
    p.add_argument("--ct-dir",         type=str, default=None,
                   help="[nii/jitter_nii] Katalog z plikami CT *_0000.nii.gz")
    p.add_argument("--seg-dir",        type=str, required=True,
                   help="Katalog z plikami segmentacji "
                        "([nii] *_fullseg.nii.gz | [body_raw] body{N}_img{M}.nii.gz | "
                        "[jitter_nii] jitter_Clean*_slice*.nii.gz bez _0000)")
    # -- tryb body_raw --
    p.add_argument("--data-root",      type=str, default=None,
                   help="[body_raw] Katalog główny datasetu body (zawiera body6/, body8/, ...)")
    p.add_argument("--body-folders",   type=str, nargs="+", default=["body6", "body8", "body9"],
                   help="[body_raw] Które foldery body przetworzyć")
    # -- wspólne --
    p.add_argument("--out-dir",        type=str, required=True,
                   help="Katalog wyjściowy dla seg_channel_img*.raw")
    p.add_argument("--num-workers",    type=int, default=4)
    p.add_argument("--limit",          type=int, default=None,
                   help="Ogranicz liczbę obrazów (do testów)")
    p.add_argument("--T-body",         type=float, default=-400.0)
    p.add_argument("--r-close",        type=int,   default=5)
    p.add_argument("--min-area-organ", type=int,   default=200)
    p.add_argument("--min-area-other", type=int,   default=500)
    p.add_argument("--flare-L",        type=int,   default=5)
    p.add_argument("--flare-K",        type=int,   default=6)
    p.add_argument("--hop-decay",      type=float, default=0.5,
                   help="Dyskonto sygnału na każdy hop BFS w 2D (0..1)")
    p.add_argument("--max-hops",       type=int,   default=10)
    p.add_argument("--z-neighbors",    type=int,   default=2,
                   help="Ile sąsiednich klatek Z wziąć pod uwagę (0 = wyłączone)")
    p.add_argument("--z-decay",        type=float, default=0.7,
                   help="Dyskonto per klatka Z (0..1); z_weight = z_decay^|dz|")
    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "T_body": args.T_body, "r_close": args.r_close,
        "min_area_organ": args.min_area_organ, "min_area_other": args.min_area_other,
        "flare_L": args.flare_L, "flare_K": args.flare_K,
        "hop_decay": args.hop_decay, "max_hops": args.max_hops,
    }

    # --- Indeksowanie zależne od trybu ---
    if args.mode == "nii":
        if not args.ct_dir:
            raise ValueError("--ct-dir wymagane w trybie nii")
        ct_map, seg_map, common = index_files(args.ct_dir, args.seg_dir)
        worker_fn = process_one
    elif args.mode == "body_raw":
        if not args.data_root:
            raise ValueError("--data-root wymagane w trybie body_raw")
        ct_map, seg_map, common = index_files_body_raw(
            args.data_root, args.seg_dir, args.body_folders
        )
        worker_fn = process_one_body_raw
    elif args.mode == "jitter_nii":
        if not args.ct_dir:
            raise ValueError("--ct-dir wymagane w trybie jitter_nii")
        ct_map, seg_map, common = index_files_jitter_nii(args.ct_dir, args.seg_dir)
        worker_fn = process_one_jitter_nii
    else:  # real_nii
        if not args.ct_dir:
            raise ValueError("--ct-dir wymagane w trybie real_nii")
        ct_map, seg_map, common = index_files_real_nii(args.ct_dir, args.seg_dir)
        worker_fn = process_one_jitter_nii   # identyczna logika przetwarzania

    if args.limit:
        common = common[:args.limit]

    print(f"Tryb: {args.mode}")
    if args.mode == "body_raw":
        print(f"Foldery body: {args.body_folders}")
    print(f"Obrazów do przetworzenia: {len(common)}")
    print(f"Sąsiedzi Z: ±{args.z_neighbors}  z_decay={args.z_decay}")

    tasks = []
    for iid in common:
        body_id, ct_path  = ct_map[iid]
        _,       seg_path = seg_map[iid]
        neighbor_specs = get_neighbor_seg_specs(
            iid, body_id, seg_map,
            n_neighbors=args.z_neighbors,
            z_decay=args.z_decay,
        )
        tasks.append((iid, body_id, ct_path, seg_path, neighbor_specs, str(out_dir), cfg))

    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(worker_fn, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            iid, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  [ERR] img{iid}: {status}")
            if i % 100 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={ok} skip={skip} err={err}")

    print(f"\nGotowe. ok={ok} skip={skip} err={err}")
    print(f"Kanały zapisane w: {out_dir}")


if __name__ == "__main__":
    main()
