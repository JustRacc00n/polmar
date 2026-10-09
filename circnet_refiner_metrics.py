"""
circnet_refiner_metrics.py — Metryki UNet refinera (Baseline → Prior → Refiner)

Dla każdego obrazu ze zbioru testowego liczy (poza strefą metalu):
  MAE, RMSE, PSNR, SSIM, gradient MAE
  dla trzech punktów: oryginał (art), Prior (CircNet lub NMAR), Refiner

Uruchomienie (CircNet):
    python3 circnet_refiner_metrics.py \
        --model-path  /app/CircNet/refiner_run_XXXXX/best_refiner.pth \
        --data-root   /app/data/RPI/body \
        --prior-dir   /app/CircNet/mar_run_XXXXX/corrected \
        --test-folder body5 \
        --output-dir  /app/CircNet/refiner_run_XXXXX/metrics

Uruchomienie (NMAR):
    python3 circnet_refiner_metrics.py \
        --model-path  /app/CircNet/refiner_run_XXXXX/best_refiner.pth \
        --data-root   /app/data/RPI/body \
        --prior-dir   /app/output/aapm_inference_results \
        --prior-suffix _nmar2 \
        --test-folder body5 \
        --output-dir  /app/CircNet/refiner_run_XXXXX/metrics

Opcjonalnie z kanałem segmentacji:
        --seg-channel-dir /app/CircNet/seg_channels
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from skimage.metrics import structural_similarity as compute_ssim

from dataset import load_raw, find_subfolders, normalize, denormalize, HU_MIN, HU_MAX
from circnet_refiner_train import (
    IDX_PAT, WIN_MIN, WIN_MAX,
    UNet, forward_refine, build_samples,
)

# -------------------------------------------------------------------------
# Metryki
# -------------------------------------------------------------------------
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
                         angle_label: str = 'Prior (Angle)') -> None:
    """
    Zapisuje porównania PNG. Kolumny dobierane dynamicznie:

    base (no_prior):        Artefakt | Refiner | Referencja
    single prior:           Artefakt | Prior | Refiner | Referencja
    two priors (+ angle):   Artefakt | Prior | Prior (Angle) | Refiner | Referencja

    Wybiera n_samples obrazów o najwyższym mae_art (najtrudniejsze przypadki).
    Okienkowanie: soft tissue C=40 W=400 HU → [-160, 240].
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

    # Buduj listę kolumn dynamicznie
    img_keys = ['_art_hu']
    labels   = ['Artefakt (art)']
    if not no_prior:
        img_keys.append('_prior_hu')
        labels.append(prior_label)
    if has_angle:
        img_keys.append('_angle_hu')
        labels.append(angle_label)
    img_keys += ['_pred_hu', '_ref_hu']
    labels   += ['Refiner', 'Referencja (ref)']

    n_cols  = len(labels)
    fig_w   = 4 * n_cols
    grid_title = 'Porównanie: ' + ' | '.join(labels)

    fig_grid, axes_grid = plt.subplots(n_sel, n_cols,
                                        figsize=(fig_w, 4 * n_sel),
                                        squeeze=False)
    fig_grid.suptitle(grid_title, fontsize=12, y=1.001)

    for gi, row in enumerate(selected):
        iid    = row['iid']
        imgs_w = [w01(row[k]) for k in img_keys]

        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 4))
        for ax, img_w, lbl in zip(axes, imgs_w, labels):
            ax.imshow(img_w, cmap='gray', vmin=0, vmax=1)
            ax.set_title(lbl, fontsize=9)
            ax.axis('off')
        fig.suptitle(f'img{iid}  |  MAE art→pred: {row["mae_art"]:.1f}→{row["mae_pred"]:.1f} HU  '
                     f'PSNR: {row["psnr_art"]:.2f}→{row["psnr_pred"]:.2f} dB  '
                     f'SSIM: {row["ssim_art"]:.3f}→{row["ssim_pred"]:.3f}',
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(png_dir / f'img{iid:04d}_comparison.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        for col, (img_w, lbl) in enumerate(zip(imgs_w, labels)):
            axes_grid[gi][col].imshow(img_w, cmap='gray', vmin=0, vmax=1)
            axes_grid[gi][col].axis('off')
            if gi == 0:
                axes_grid[gi][col].set_title(lbl, fontsize=10, fontweight='bold')
        axes_grid[gi][0].set_ylabel(f'img{iid}', rotation=0, labelpad=50, va='center', fontsize=8)

    fig_grid.tight_layout()
    fig_grid.savefig(output_dir / 'comparison_grid.png', dpi=100, bbox_inches='tight')
    plt.close(fig_grid)
    print(f'PNGi porównawcze: {png_dir}  (grid: {output_dir / "comparison_grid.png"})')


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path',      type=str, required=True)
    p.add_argument('--data-root',       type=str, required=True)
    p.add_argument('--prior-dir',       type=str, default=None,
                   help='Katalog z plikami prior. Pomiń przy --no-prior.')
    p.add_argument('--prior-suffix',    type=str, default='',
                   help='Sufiks pliku prior, np. "_nmar2" dla NMAR')
    p.add_argument('--no-prior',        action='store_true', default=False,
                   help='Tryb bez prioru (musi zgadzać się z trenowanym modelem)')
    p.add_argument('--angle-dir',       type=str, default=None,
                   help='Drugi prior jako dodatkowy kanał')
    p.add_argument('--angle-suffix',    type=str, default='',
                   help='Sufiks pliku angle-prior')
    p.add_argument('--seg-channel-dir', type=str, default=None,
                   help='Opcjonalny katalog z kanałem segmentacji (seg_channel_imgN_512x512x1.raw)')
    p.add_argument('--test-folder',     type=str, required=True)
    p.add_argument('--output-dir',      type=str, required=True)
    p.add_argument('--base-features',   type=int, default=32,
                   help='Używane tylko gdy brak info w checkpoincie')
    p.add_argument('--no-hard-paste',   action='store_true', default=False)
    p.add_argument('--device',          type=str, default='cuda')
    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    hard_paste = not args.no_hard_paste

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Załaduj model ---
    ckpt = torch.load(args.model_path, map_location=device)

    # Wykryj liczbę kanałów wejściowych z zapisanych argumentów treningu
    saved_args = ckpt.get('args', {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    no_prior  = saved_args.get('no_prior', False)
    base_art  = saved_args.get('base_art', False)
    has_angle = bool(saved_args.get('angle_dir', None))
    has_seg   = bool(saved_args.get('seg_channel_dir', None))
    in_channels = 1 + (0 if no_prior else 1) + (1 if has_angle else 0) + (1 if has_seg else 0)
    base_ch   = 0 if (no_prior or base_art) else 1
    base_features = saved_args.get('base_features', args.base_features)

    model = UNet(in_ch=in_channels, out_ch=1, base=base_features).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Model: {args.model_path}  (ep={ckpt.get("epoch","?")}  '
          f'in_ch={in_channels}  base={base_features}  no_prior={no_prior}')

    # --- Zbierz próbki testowe ---
    data_root      = Path(args.data_root)
    prior_dir      = Path(args.prior_dir) if args.prior_dir else None
    prior_suffix   = args.prior_suffix
    angle_dir      = Path(args.angle_dir) if args.angle_dir else None
    seg_channel_dir = Path(args.seg_channel_dir) if args.seg_channel_dir else None

    samples = build_samples(data_root, prior_dir, [args.test_folder],
                            prior_suffix=prior_suffix, no_prior=no_prior,
                            angle_dir=angle_dir, angle_suffix=args.angle_suffix)
    print(f'Obrazów testowych: {len(samples)}  (folder: {args.test_folder})')
    if prior_suffix:
        print(f'Prior: {prior_dir}  (sufiks: "{prior_suffix}")')
    else:
        print(f'Prior: {prior_dir}  (CircNet)')
    if seg_channel_dir:
        print(f'Seg channel dir: {seg_channel_dir}')

    rows = []
    for art_path, prior_path, tgt_path, msk_path, angle_path in samples:
        iid = int(IDX_PAT.search(art_path.name).group(1))

        art  = np.clip(load_raw(art_path), HU_MIN, HU_MAX)
        tgt  = np.clip(load_raw(tgt_path),  HU_MIN, HU_MAX)
        msk  = load_raw(msk_path)

        metal_mask = (msk > 0.5).astype(np.float32)
        outmask    = 1.0 - metal_mask

        # Referencja: clean poza metalem, art na metalu
        ref_hu = (1.0 - metal_mask) * tgt + metal_mask * art

        # Prior
        if not no_prior and prior_path is not None:
            prior = np.clip(load_raw(prior_path), HU_MIN, HU_MAX)
        else:
            prior = art  # placeholder do metryk (art→art = 0 gain)

        # Angle prior (do PNG i inferencji)
        angle_hu = None
        if has_angle and angle_path is not None:
            angle_hu = np.clip(load_raw(angle_path), HU_MIN, HU_MAX)

        # --- Inferencja refinera ---
        art_n    = normalize(art)
        channels = [art_n]
        if not no_prior:
            channels.append(normalize(prior))
        if angle_hu is not None:
            channels.append(normalize(angle_hu))
        if has_seg and seg_channel_dir is not None:
            seg_path = seg_channel_dir / f'seg_channel_img{iid}_512x512x1.raw'
            seg_ch = load_raw(seg_path).clip(0.0, 1.0) if seg_path.exists() else np.zeros_like(art_n)
            channels.append(seg_ch)

        x_in = np.stack(channels, axis=0)[np.newaxis]   # [1, C, H, W]
        m_in = metal_mask[np.newaxis, np.newaxis]         # [1, 1, H, W]

        x_t = torch.from_numpy(x_in).float().to(device)
        m_t = torch.from_numpy(m_in).float().to(device)

        with torch.no_grad():
            pred_n = forward_refine(model, x_t, m_t, hard_paste=hard_paste, base_ch=base_ch)
        pred_n  = pred_n.squeeze().cpu().numpy()
        pred_hu = denormalize(pred_n)

        # --- Windowed [0,1] ---
        art_w   = window01(art)
        prior_w = window01(prior)
        ref_w   = window01(ref_hu)
        pred_w  = window01(pred_hu)
        angle_w = window01(angle_hu) if angle_hu is not None else None

        # --- Metryki ---
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
        grad_art   = masked_mae(sobel_mag(art_w),   g_ref, outmask)
        grad_prior = masked_mae(sobel_mag(prior_w), g_ref, outmask)
        grad_pred  = masked_mae(sobel_mag(pred_w),  g_ref, outmask)

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
            'iid':         iid,
            'mae_art':     mae_art,    'mae_prior':  mae_prior,   'mae_pred':  mae_pred,
            'rmse_art':    rmse_art,   'rmse_prior': rmse_prior,  'rmse_pred': rmse_pred,
            'psnr_art':    psnr_art,   'psnr_prior': psnr_prior,  'psnr_pred': psnr_pred,
            'ssim_art':    ssim_art,   'ssim_prior': ssim_prior,  'ssim_pred': ssim_pred,
            'grad_art':    grad_art,   'grad_prior': grad_prior,  'grad_pred': grad_pred,
            # surowe obrazy do PNG (zostaną usunięte przed CSV)
            '_art_hu': art, '_prior_hu': prior, '_pred_hu': pred_hu, '_ref_hu': ref_hu,
        }
        if angle_hu is not None:
            row['_angle_hu']  = angle_hu
            row['mae_angle']  = mae_angle
            row['rmse_angle'] = rmse_angle
            row['psnr_angle'] = psnr_angle
            row['ssim_angle'] = ssim_angle
            row['grad_angle'] = grad_angle
        rows.append(row)

        angle_info = (f' → {mae_angle:.1f}(ang)' if mae_angle is not None else '')
        print(f'  #{iid:4d}  '
              f'MAE: {mae_art:.1f} → {mae_prior:.1f}{angle_info} → {mae_pred:.1f} HU  '
              f'PSNR: {psnr_art:.2f} → {psnr_prior:.2f} → {psnr_pred:.2f} dB  '
              f'SSIM: {ssim_art:.3f} → {ssim_prior:.3f} → {ssim_pred:.3f}')

    # Przed DataFrame — wyodrębnij surowe obrazy i wygeneruj PNGi
    prior_label_png = 'Prior (NMAR)' if prior_suffix else 'Prior (CircNet)'
    angle_label_png = 'Prior (CircNet)' if has_angle else 'Prior (Angle)'
    save_comparison_pngs(rows, output_dir, n_samples=10,
                         no_prior=no_prior, prior_label=prior_label_png,
                         has_angle=has_angle, angle_label=angle_label_png)
    # Usuń klucze surowych obrazów przed budowaniem DataFrame
    for r in rows:
        r.pop('_art_hu', None); r.pop('_prior_hu', None)
        r.pop('_pred_hu', None); r.pop('_ref_hu', None)
        r.pop('_angle_hu', None)

    df = pd.DataFrame(rows)

    # Gain vs oryginał (baseline)  —  + = lepiej niż art
    df['mae_gain_prior']  = df['mae_art']    - df['mae_prior']
    df['mae_gain_pred']   = df['mae_art']    - df['mae_pred']
    df['rmse_gain_prior'] = df['rmse_art']   - df['rmse_prior']
    df['rmse_gain_pred']  = df['rmse_art']   - df['rmse_pred']
    df['psnr_gain_prior'] = df['psnr_prior'] - df['psnr_art']
    df['psnr_gain_pred']  = df['psnr_pred']  - df['psnr_art']
    df['ssim_gain_prior'] = df['ssim_prior'] - df['ssim_art']
    df['ssim_gain_pred']  = df['ssim_pred']  - df['ssim_art']
    df['grad_gain_prior'] = df['grad_art']   - df['grad_prior']
    df['grad_gain_pred']  = df['grad_art']   - df['grad_pred']
    if has_angle and 'mae_angle' in df.columns:
        df['mae_gain_angle']  = df['mae_art']    - df['mae_angle']
        df['rmse_gain_angle'] = df['rmse_art']   - df['rmse_angle']
        df['psnr_gain_angle'] = df['psnr_angle'] - df['psnr_art']
        df['ssim_gain_angle'] = df['ssim_angle'] - df['ssim_art']
        df['grad_gain_angle'] = df['grad_art']   - df['grad_angle']

    # Dodatkowy gain Refiner vs Prior
    df['mae_gain_ref_vs_prior']  = df['mae_prior']  - df['mae_pred']
    df['psnr_gain_ref_vs_prior'] = df['psnr_pred']  - df['psnr_prior']
    df['ssim_gain_ref_vs_prior'] = df['ssim_pred']  - df['ssim_prior']

    prior_label  = f'prior({prior_suffix or "CircNet"})'
    angle_label2 = 'prior(angle)' if has_angle else None
    csv_path = output_dir / f'metrics_refiner_{args.test_folder}.csv'
    df.to_csv(csv_path, index=False)

    print(f'\n{"="*60}')
    print(f'=== Poprawa vs oryginał (Baseline) ===')
    hdr = f'{"Metryka":<14}  {prior_label:>16}'
    if angle_label2:
        hdr += f'  {angle_label2:>14}'
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
            line += f'  {df[a_col].mean():+14.4f}'
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
        hdr2 += f'  {angle_label2:>16}'
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
        'test_folder': args.test_folder,
        'n_images': len(df),
        'prior': prior_suffix or 'CircNet',
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
