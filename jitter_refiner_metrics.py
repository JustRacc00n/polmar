"""
jitter_refiner_metrics.py — Metryki UNet refinera dla datasetu szpitalnego (jitter)

Dla każdego obrazu ze zbioru testowego liczy (poza strefą metalu):
  MAE, RMSE, PSNR, SSIM, gradient MAE
  dla trzech punktów: oryginał (art), NMAR prior, Refiner

Uruchomienie:
    python3 jitter_refiner_metrics.py \
        --model-path   /app/CircNet/refiner_jitter_run_XXXXX/best_refiner.pth \
        --art-dir      /app/output/3frames_jitter_raw_clean_real \
        --clean-dir    /app/data/Clean_raw \
        --nmar-dir     /app/output/nmar_results \
        --seg-chan-dir /app/data/seg_channels_jitter \
        --test-scan-id 4 \
        --output-dir   /app/CircNet/refiner_jitter_run_XXXXX/metrics
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from skimage.metrics import structural_similarity as compute_ssim

from jitter_refiner_train import (
    ART_RE, METAL_HU_DEFAULT, IMG_SIZE,
    load_raw, normalize,
    UNet, forward_refine, build_samples_jitter,
)

# -------------------------------------------------------------------------
# Metryki
# -------------------------------------------------------------------------
WIN_MIN, WIN_MAX = -160.0, 240.0   # soft tissue: C=40 W=400


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
    _, ssim_map = compute_ssim(a, b, data_range=data_range, full=True)
    return (ssim_map * mask).sum() / (mask.sum() + 1e-8)


def sobel_mag(img: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(img); gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    return np.sqrt(gx * gx + gy * gy)



# -------------------------------------------------------------------------
# Generowanie obrazków porównawczych (PNG)
# -------------------------------------------------------------------------
def save_comparison_pngs(rows_data: list, output_dir: 'Path',
                         n_samples: int = 10, no_prior: bool = False,
                         prior_label: str = 'Prior', has_angle: bool = False,
                         angle_label: str = 'Prior (Angle)',
                         has_seg: bool = False,
                         use_3frames: bool = False) -> None:
    """
    Zapisuje porównania PNG. Kolumny dobierane dynamicznie:

    base (no_prior):             Artefakt | Refiner | Referencja
    base + seg:                  Artefakt | Seg (graf) | Refiner | Referencja
    single prior:                Artefakt | Prior | Refiner | Referencja
    single prior + seg:          Artefakt | Prior | Seg (graf) | Refiner | Referencja
    two priors (+ angle):        Artefakt | Prior | Prior (Angle) | Refiner | Referencja
    two priors + seg:            Artefakt | Prior | Prior (Angle) | Seg (graf) | Refiner | Referencja
    3f-base:                     Klatka -1 | Klatka 0 | Klatka +1 | Refiner | Referencja
    3f-circnet:                  CircNet -1 | CircNet 0 | CircNet +1 | Refiner | Referencja

    Okienkowanie HU: soft tissue C=40 W=400 → [-160, 240].
    Kanał seg: wyświetlany jako mapa prawdopodobieństwa [0,1] (bez okienkowania HU).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[WARN] matplotlib niedostępny, pomijam PNGi: {e}')
        return

    W_MIN, W_MAX = WIN_MIN, WIN_MAX

    def w01(img):
        return (np.clip(img, W_MIN, W_MAX) - W_MIN) / (W_MAX - W_MIN)

    rows_sorted = sorted(rows_data, key=lambda r: r['mae_art'], reverse=True)
    selected = rows_sorted[:n_samples]
    n_sel = len(selected)

    png_dir = output_dir / 'comparison_pngs'
    png_dir.mkdir(exist_ok=True)

    # ---- Układ kolumn ----
    if use_3frames:
        # Warianty 3-klatkowe: pokaż wszystkie 3 wejścia sieci
        if no_prior:
            # 3f-base: [art_prev, art_curr, art_next]
            img_keys = ['_frame_prev_hu', '_art_hu', '_frame_next_hu', '_pred_hu', '_ref_hu']
            labels   = ['Klatka -1', 'Klatka 0 (art)', 'Klatka +1', 'Refiner', 'Referencja (ref)']
        else:
            # 3f-circnet: [circ_prev, circ_curr, circ_next]
            img_keys = ['_frame_prev_hu', '_prior_hu', '_frame_next_hu', '_pred_hu', '_ref_hu']
            labels   = [f'{prior_label} -1', prior_label, f'{prior_label} +1', 'Refiner', 'Referencja (ref)']
        modes = ['hu', 'hu', 'hu', 'hu', 'hu']
    else:
        img_keys  = ['_art_hu']
        labels    = ['Artefakt (art)']
        # 'hu' = okienkowanie HU, 'seg' = mapa organów (tab10), 'norm' = normalizacja [0,1]
        modes     = ['hu']
        if not no_prior:
            img_keys.append('_prior_hu')
            labels.append(prior_label)
            modes.append('hu')
        if has_angle:
            img_keys.append('_angle_hu')
            labels.append(angle_label)
            modes.append('hu')
        if has_seg:
            img_keys.append('_seg')
            labels.append('Seg (organy)')
            modes.append('seg')
        img_keys += ['_pred_hu', '_ref_hu']
        labels   += ['Refiner', 'Referencja (ref)']
        modes    += ['hu', 'hu']

    n_cols     = len(labels)
    fig_w      = 4 * n_cols
    grid_title = 'Porównanie: ' + ' | '.join(labels)

    fig_grid, axes_grid = plt.subplots(n_sel, n_cols,
                                        figsize=(fig_w, 4 * n_sel),
                                        squeeze=False)
    fig_grid.suptitle(grid_title, fontsize=12, y=1.001)

    def render(ax, img, mode):
        if mode == 'hu':
            ax.imshow(w01(img), cmap='gray', vmin=0, vmax=1)
        elif mode == 'seg':
            # Mapa wieloklasowa — znormalizuj do [0,1] i użyj tab10
            seg_min, seg_max = img.min(), img.max()
            seg_norm = (img - seg_min) / (seg_max - seg_min + 1e-8)
            ax.imshow(seg_norm, cmap='tab10', vmin=0, vmax=1, interpolation='nearest')

    for gi, row in enumerate(selected):
        scan_id  = row['scan_id']
        slice_id = row['slice_id']

        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 4))
        for ax, key, lbl, mode in zip(axes, img_keys, labels, modes):
            render(ax, row[key], mode)
            ax.set_title(lbl, fontsize=9)
            ax.axis('off')
        fig.suptitle(f'scan{scan_id:02d}_sl{slice_id:04d}  |  '
                     f'MAE art→pred: {row["mae_art"]:.1f}→{row["mae_pred"]:.1f} HU  '
                     f'PSNR: {row["psnr_art"]:.2f}→{row["psnr_pred"]:.2f} dB  '
                     f'SSIM: {row["ssim_art"]:.3f}→{row["ssim_pred"]:.3f}',
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(png_dir / f'scan{scan_id:02d}_sl{slice_id:04d}_comparison.png',
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

        for col, (key, lbl, mode) in enumerate(zip(img_keys, labels, modes)):
            render(axes_grid[gi][col], row[key], mode)
            axes_grid[gi][col].axis('off')
            if gi == 0:
                axes_grid[gi][col].set_title(lbl, fontsize=10, fontweight='bold')
        axes_grid[gi][0].set_ylabel(f'sc{scan_id}_sl{slice_id}',
                                     rotation=0, labelpad=55, va='center', fontsize=7)

    fig_grid.tight_layout()
    fig_grid.savefig(output_dir / 'comparison_grid.png', dpi=100, bbox_inches='tight')
    plt.close(fig_grid)
    print(f'PNGi porównawcze: {png_dir}  (grid: {output_dir / "comparison_grid.png"})')


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',    type=str, required=True)
    p.add_argument('--art-dir',       type=str, required=True,
                   help='Katalog z plikami jitter_Clean*_slice*_H512_W512.raw')
    p.add_argument('--clean-dir',     type=str, required=True,
                   help='Katalog z plikami Clean*_slice*_H512_W512.raw')
    p.add_argument('--prior-dir',     type=str, default=None,
                   help='Katalog z plikami prior. Pomiń przy --no-prior.')
    p.add_argument('--prior-suffix',  type=str, default='_nmar2',
                   help='Sufiks pliku prior (domyślnie _nmar2; dla CircNet: _circnet)')
    p.add_argument('--no-prior',      action='store_true', default=False,
                   help='Tryb bez prioru (musi zgadzać się z trenowanym modelem)')
    p.add_argument('--angle-dir',     type=str, default=None,
                   help='Drugi prior jako dodatkowy kanał')
    p.add_argument('--angle-suffix',  type=str, default='_circnet',
                   help='Sufiks pliku angle-prior')
    p.add_argument('--seg-chan-dir',  type=str, default=None,
                   help='Katalog z kanałami semantycznymi (*_seg_channel.raw)')
    p.add_argument('--test-scan-id',  type=int, required=True,
                   help='Numer scanu do ewaluacji (np. 4)')
    p.add_argument('--output-dir',    type=str, required=True)
    p.add_argument('--metal-hu',      type=float, default=METAL_HU_DEFAULT,
                   help='Próg HU maski metalu (domyślnie 2500)')
    p.add_argument('--base-features', type=int, default=32,
                   help='Używane tylko gdy brak info w checkpoincie')
    p.add_argument('--no-hard-paste', action='store_true', default=False)
    p.add_argument('--device',        type=str, default='cuda')
    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args       = parse_args()
    device     = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    hard_paste = not args.no_hard_paste

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Załaduj model ---
    ckpt = torch.load(args.model_path, map_location=device)

    saved_args    = ckpt.get('args', {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    no_prior      = saved_args.get('no_prior', False)
    has_angle     = bool(saved_args.get('angle_dir', None))
    has_seg       = bool(saved_args.get('seg_chan_dir', None))
    use_3frames   = bool(saved_args.get('use_3frames', False))
    if use_3frames:
        in_channels = 3
        base_ch     = 1
        art_ch      = ckpt.get('art_ch', 1)
    else:
        in_channels = (1
                       + (0 if no_prior else 1)
                       + (1 if has_angle else 0)
                       + (1 if has_seg else 0))
        base_ch     = 0 if no_prior else 1
        art_ch      = ckpt.get('art_ch', 0)
    base_features = saved_args.get('base_features', args.base_features)

    model = UNet(in_ch=in_channels, out_ch=1, base=base_features).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Model: {args.model_path}  (ep={ckpt.get("epoch","?")}  '
          f'in_ch={in_channels}  base={base_features})')

    # --- Zbierz próbki testowe ---
    art_dir      = Path(args.art_dir)
    clean_dir    = Path(args.clean_dir)
    prior_dir    = Path(args.prior_dir) if args.prior_dir else None
    angle_dir    = Path(args.angle_dir) if args.angle_dir else None
    seg_chan_dir  = Path(args.seg_chan_dir) if args.seg_chan_dir else None

    samples = build_samples_jitter(
        art_dir, clean_dir, prior_dir, seg_chan_dir,
        scan_ids=[args.test_scan_id],
        prior_suffix=args.prior_suffix,
        no_prior=no_prior,
        angle_dir=angle_dir, angle_suffix=args.angle_suffix,
        use_3frames=use_3frames,
    )
    prior_label = 'CircNet' if args.prior_suffix == '_circnet' else f'prior({args.prior_suffix})'
    print(f'Obrazów testowych: {len(samples)}  (scan: {args.test_scan_id}  prior: {prior_label}  3frames: {use_3frames})')

    rows = []
    for sample in samples:
        art_p, clean_p, prior_p, seg_p, angle_p = sample[:5]
        art_prev_p = sample[5] if len(sample) > 5 else None
        art_next_p = sample[6] if len(sample) > 6 else None
        mt = ART_RE.search(art_p.name)
        scan_id  = int(mt.group('scan'))
        slice_id = int(mt.group('slice'))

        art   = load_raw(art_p)
        clean = load_raw(clean_p)

        metal_mask = (art > args.metal_hu).astype(np.float32)
        outmask    = 1.0 - metal_mask

        # Referencja: clean poza metalem, art na metalu
        ref_hu = (1.0 - metal_mask) * clean + metal_mask * art

        # Prior
        if not no_prior and prior_p is not None:
            prior = load_raw(prior_p)
        else:
            prior = art  # placeholder

        # Angle prior (do PNG i inferencji)
        angle_hu = None
        if has_angle and angle_p is not None:
            angle_hu = load_raw(angle_p)

        # --- Inferencja refinera ---
        art_n    = normalize(art)
        channels = [art_n]
        if not no_prior:
            channels.append(normalize(prior))
        if angle_hu is not None:
            channels.append(normalize(angle_hu))
        seg_ch_img = None
        if has_seg and seg_chan_dir is not None:
            if seg_p is not None and seg_p.exists():
                seg_ch_img = load_raw(seg_p).clip(0.0, 1.0)
            else:
                seg_ch_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            channels.append(seg_ch_img)

        if use_3frames:
            frame_prev = load_raw(art_prev_p) if art_prev_p is not None else art
            frame_next = load_raw(art_next_p) if art_next_p is not None else art
            if no_prior:
                # 3f-base: [art_prev, art_curr, art_next]
                channels = [normalize(frame_prev), normalize(art), normalize(frame_next)]
            else:
                # 3f-circnet: [circ_prev, circ_curr, circ_next]
                prior_n = normalize(prior)
                channels = [normalize(frame_prev), prior_n, normalize(frame_next)]
            # Zapamiętaj klatki sąsiednie do wizualizacji
            _frame_prev_hu = frame_prev
            _frame_next_hu = frame_next
        else:
            _frame_prev_hu = None
            _frame_next_hu = None

        x_in = np.stack(channels, axis=0)[np.newaxis]   # [1, C, H, W]
        m_in = metal_mask[np.newaxis, np.newaxis]         # [1, 1, H, W]

        x_t = torch.from_numpy(x_in).float().to(device)
        m_t = torch.from_numpy(m_in).float().to(device)

        with torch.no_grad():
            pred_n = forward_refine(model, x_t, m_t, hard_paste=hard_paste,
                                    base_ch=base_ch, art_ch=art_ch)
        pred_n  = pred_n.squeeze().cpu().numpy()
        # Denormalizuj: HU = pred_n * (HU_MAX - HU_MIN) + HU_MIN
        pred_hu = pred_n * (1000.0 - (-1000.0)) + (-1000.0)

        # Windowed [0,1]
        art_w   = window01(art)
        prior_w = window01(prior)
        ref_w   = window01(ref_hu)
        pred_w  = window01(pred_hu)
        angle_w = window01(angle_hu) if angle_hu is not None else None

        mae_art   = masked_mae(art,     ref_hu, outmask)
        mae_prior = masked_mae(prior,   ref_hu, outmask)
        mae_pred  = masked_mae(pred_hu, ref_hu, outmask)

        rmse_art   = masked_rmse(art,     ref_hu, outmask)
        rmse_prior = masked_rmse(prior,   ref_hu, outmask)
        rmse_pred  = masked_rmse(pred_hu, ref_hu, outmask)

        psnr_art   = masked_psnr(art_w,   ref_w, outmask)
        psnr_prior = masked_psnr(prior_w, ref_w, outmask)
        psnr_pred  = masked_psnr(pred_w,  ref_w, outmask)

        ssim_art   = masked_ssim(art_w,   ref_w, outmask)
        ssim_prior = masked_ssim(prior_w, ref_w, outmask)
        ssim_pred  = masked_ssim(pred_w,  ref_w, outmask)

        g_ref      = sobel_mag(ref_w)
        grad_art   = masked_mae(sobel_mag(art_w),  g_ref, outmask)
        grad_prior = masked_mae(sobel_mag(prior_w), g_ref, outmask)
        grad_pred  = masked_mae(sobel_mag(pred_w), g_ref, outmask)

        # Metryki dla drugiego prioru (angle), jeśli istnieje
        if angle_w is not None:
            mae_angle  = masked_mae(angle_hu, ref_hu, outmask)
            rmse_angle = masked_rmse(angle_hu, ref_hu, outmask)
            psnr_angle = masked_psnr(angle_w,  ref_w,  outmask)
            ssim_angle = masked_ssim(angle_w,  ref_w,  outmask)
            grad_angle = masked_mae(sobel_mag(angle_w), g_ref, outmask)
        else:
            mae_angle = rmse_angle = psnr_angle = ssim_angle = grad_angle = None

        row = {
            'scan_id': scan_id, 'slice_id': slice_id,
            'mae_art':  mae_art,   'mae_prior':  mae_prior,  'mae_pred':  mae_pred,
            'rmse_art': rmse_art,  'rmse_prior': rmse_prior, 'rmse_pred': rmse_pred,
            'psnr_art': psnr_art,  'psnr_prior': psnr_prior, 'psnr_pred': psnr_pred,
            'ssim_art': ssim_art,  'ssim_prior': ssim_prior, 'ssim_pred': ssim_pred,
            'grad_art': grad_art,  'grad_prior': grad_prior, 'grad_pred': grad_pred,
            '_art_hu': art, '_prior_hu': prior, '_pred_hu': pred_hu, '_ref_hu': ref_hu,
        }
        if _frame_prev_hu is not None:
            row['_frame_prev_hu'] = _frame_prev_hu
            row['_frame_next_hu'] = _frame_next_hu
        if angle_hu is not None:
            row['_angle_hu']  = angle_hu
            row['mae_angle']  = mae_angle
            row['rmse_angle'] = rmse_angle
            row['psnr_angle'] = psnr_angle
            row['ssim_angle'] = ssim_angle
            row['grad_angle'] = grad_angle
        if seg_ch_img is not None:
            row['_seg'] = seg_ch_img
        rows.append(row)

        angle_info = (f' → {mae_angle:.1f}(ang)' if mae_angle is not None else '')
        print(f'  scan{scan_id:02d}_sl{slice_id:04d}  '
              f'MAE: {mae_art:.1f} → {mae_prior:.1f}{angle_info} → {mae_pred:.1f} HU  '
              f'PSNR: {psnr_art:.2f} → {psnr_prior:.2f} → {psnr_pred:.2f} dB  '
              f'SSIM: {ssim_art:.3f} → {ssim_prior:.3f} → {ssim_pred:.3f}')

    prior_label_png = ('Prior (NMAR)' if args.prior_suffix == '_nmar2'
                       else 'Prior (CircNet)' if not no_prior else 'Prior')
    angle_label_png = 'Prior (CircNet-angle)' if has_angle else 'Prior (Angle)'
    save_comparison_pngs(rows, output_dir, n_samples=10,
                         no_prior=no_prior, prior_label=prior_label_png,
                         has_angle=has_angle, angle_label=angle_label_png,
                         has_seg=has_seg, use_3frames=use_3frames)
    for r in rows:
        r.pop('_art_hu', None); r.pop('_prior_hu', None)
        r.pop('_pred_hu', None); r.pop('_ref_hu', None)
        r.pop('_angle_hu', None); r.pop('_seg', None)
        r.pop('_frame_prev_hu', None); r.pop('_frame_next_hu', None)

    df = pd.DataFrame(rows)

    df['mae_gain_prior']      = df['mae_art']    - df['mae_prior']
    df['mae_gain_pred']       = df['mae_art']    - df['mae_pred']
    df['rmse_gain_prior']     = df['rmse_art']   - df['rmse_prior']
    df['rmse_gain_pred']      = df['rmse_art']   - df['rmse_pred']
    df['psnr_gain_prior']     = df['psnr_prior'] - df['psnr_art']
    df['psnr_gain_pred']      = df['psnr_pred']  - df['psnr_art']
    df['ssim_gain_prior']     = df['ssim_prior'] - df['ssim_art']
    df['ssim_gain_pred']      = df['ssim_pred']  - df['ssim_art']
    df['grad_gain_prior']     = df['grad_art']   - df['grad_prior']
    df['grad_gain_pred']      = df['grad_art']   - df['grad_pred']
    if has_angle and 'mae_angle' in df.columns:
        df['mae_gain_angle']  = df['mae_art']    - df['mae_angle']
        df['rmse_gain_angle'] = df['rmse_art']   - df['rmse_angle']
        df['psnr_gain_angle'] = df['psnr_angle'] - df['psnr_art']
        df['ssim_gain_angle'] = df['ssim_angle'] - df['ssim_art']
        df['grad_gain_angle'] = df['grad_art']   - df['grad_angle']
    df['mae_gain_ref_vs_prior']  = df['mae_prior']  - df['mae_pred']
    df['psnr_gain_ref_vs_prior'] = df['psnr_pred']  - df['psnr_prior']
    df['ssim_gain_ref_vs_prior'] = df['ssim_pred']  - df['ssim_prior']

    prior_label  = ('Prior (NMAR)' if args.prior_suffix == '_nmar2'
                    else 'Prior (CircNet)' if not no_prior else 'Prior')
    angle_label2 = 'Prior (CircNet-angle)' if has_angle else None

    csv_path = output_dir / f'metrics_refiner_scan{args.test_scan_id:02d}.csv'
    df.to_csv(csv_path, index=False)

    print(f'\n{"="*60}')
    print(f'=== Poprawa vs oryginał (art) ===')
    hdr = f'{"Metryka":<14}  {prior_label:>16}'
    if angle_label2:
        hdr += f'  {angle_label2:>22}'
    hdr += f'  {"Refiner":>10}'
    print(hdr); print('-' * len(hdr))
    for label, c_col, a_col, p_col in [
        ('MAE [HU]',  'mae_gain_prior',  'mae_gain_angle',  'mae_gain_pred'),
        ('RMSE [HU]', 'rmse_gain_prior', 'rmse_gain_angle', 'rmse_gain_pred'),
        ('PSNR [dB]', 'psnr_gain_prior', 'psnr_gain_angle', 'psnr_gain_pred'),
        ('SSIM',      'ssim_gain_prior', 'ssim_gain_angle', 'ssim_gain_pred'),
        ('GradMAE',   'grad_gain_prior', 'grad_gain_angle', 'grad_gain_pred'),
    ]:
        c = df[c_col].mean(); p = df[p_col].mean()
        line = f'{label:<14}  {c:+16.4f}'
        if angle_label2 and a_col in df.columns:
            line += f'  {df[a_col].mean():+22.4f}'
        line += f'  {p:+10.4f}'
        print(line)

    print(f'\n=== Refiner vs {prior_label} (gain > 0 = refiner lepszy) ===')
    print(f'  MAE  gain: {df["mae_gain_ref_vs_prior"].mean():+.4f} HU  '
          f'(poprawa w {(df["mae_gain_ref_vs_prior"]>0).mean()*100:.1f}% przypadków)')
    print(f'  PSNR gain: {df["psnr_gain_ref_vs_prior"].mean():+.4f} dB')
    print(f'  SSIM gain: {df["ssim_gain_ref_vs_prior"].mean():+.4f}')

    # --- Wartości absolutne ---
    print(f'\n=== Wartości absolutne (mean ± std) ===')
    hdr2 = f'{"Metryka":<14}  {"Artefakt":>14}  {prior_label:>16}'
    if angle_label2:
        hdr2 += f'  {angle_label2:>22}'
    hdr2 += f'  {"Refiner":>14}'
    print(hdr2); print('-' * len(hdr2))
    for label, a_col, c_col, ang_col, p_col in [
        ('MAE [HU]',  'mae_art',  'mae_prior',  'mae_angle',  'mae_pred'),
        ('RMSE [HU]', 'rmse_art', 'rmse_prior', 'rmse_angle', 'rmse_pred'),
        ('PSNR [dB]', 'psnr_art', 'psnr_prior', 'psnr_angle', 'psnr_pred'),
        ('SSIM',      'ssim_art', 'ssim_prior', 'ssim_angle', 'ssim_pred'),
        ('GradMAE',   'grad_art', 'grad_prior', 'grad_angle', 'grad_pred'),
    ]:
        a = df[a_col].mean(); a_s = df[a_col].std()
        c = df[c_col].mean(); c_s = df[c_col].std()
        p = df[p_col].mean(); p_s = df[p_col].std()
        line = f'{label:<14}  {a:>7.4f}±{a_s:.4f}  {c:>7.4f}±{c_s:.4f}'
        if angle_label2 and ang_col in df.columns:
            ang = df[ang_col].mean(); ang_s = df[ang_col].std()
            line += f'  {ang:>7.4f}±{ang_s:.4f}'
        line += f'  {p:>7.4f}±{p_s:.4f}'
        print(line)

    # --- Summary JSON ---
    base_cols = ['mae_art', 'mae_prior', 'mae_pred',
                 'rmse_art', 'rmse_prior', 'rmse_pred',
                 'psnr_art', 'psnr_prior', 'psnr_pred',
                 'ssim_art', 'ssim_prior', 'ssim_pred',
                 'grad_art', 'grad_prior', 'grad_pred']
    angle_cols = ['mae_angle', 'rmse_angle', 'psnr_angle',
                  'ssim_angle', 'grad_angle'] if has_angle else []
    summary = {
        'test_scan_id': args.test_scan_id,
        'n_images': len(df),
        'prior_suffix': '—' if no_prior else args.prior_suffix,
        'metrics': {}
    }
    for col in base_cols + angle_cols:
        if col in df.columns:
            summary['metrics'][col] = {'mean': float(df[col].mean()), 'std': float(df[col].std())}
    json_path = output_dir / 'summary.json'
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f'Summary JSON: {json_path}')

    print(f'\nMetryki zapisane: {csv_path}')


if __name__ == '__main__':
    main()
