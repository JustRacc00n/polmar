"""
circnet_refiner_train.py — U-Net Refiner: experiments with different inputs

The second channel (prior correction) can be any pre-corrected image:
  CircNet  → --prior-dir /app/CircNet/mar_run_XXXXX/corrected
               (files: identical name to Baseline)
  NMAR     → --prior-dir /app/output/aapm_inference_results/bodyN
               --prior-suffix _nmar2
               (files: <basename>_nmar2.raw)

Pipeline:
  Network input : [art, prior]          — 2 channels
              or  [art, prior, seg_chan] — 3 channels with semantic channel (nnU-Net graph)
  Prediction    : prior + (1 − mask) * delta   [residual on prior]
  Hard paste    : metal in output = original (Baseline)
  Target        : clean (Target) outside metal, Baseline inside metal

Usage examples:
  # CircNet, 2 channels
  python3 circnet_refiner_train.py \
      --data-root   /app/data/RPI/body \
      --prior-dir   /app/CircNet/mar_run_XXXXX/corrected \
      --train-folders body1 body2 body3 --val-folders body4 \
      --output-dir  /app/CircNet/refiner_run_XXXXX \
      --epochs 30 --batch-size 4

  # NMAR, 2 channels
  python3 circnet_refiner_train.py \
      --prior-dir   /app/output/aapm_inference_results \
      --prior-suffix _nmar2 \
      ...

  # CircNet + graph semantic channel, 3 channels
  python3 circnet_refiner_train.py \
      --prior-dir       /app/CircNet/mar_run_XXXXX/corrected \
      --seg-channel-dir /app/data/seg_channels \
      ...
"""

import argparse
import csv
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from dataset import load_raw, find_subfolders, normalize, denormalize, HU_MIN, HU_MAX

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
WIN_MIN, WIN_MAX = -160.0, 240.0   # soft tissue: C=40 W=400
IDX_PAT = re.compile(r'img(\d+)')


# -------------------------------------------------------------------------
# Sample list construction
# -------------------------------------------------------------------------
def prior_path_for(art_path: Path, prior_dir: Path, prior_suffix: str,
                   subfolder: str = '') -> Path:
    """
    Returns the path to the prior correction file.

    Supports two directory layouts:
    - Flat       : prior_dir / filename[_suffix].raw
                   (CircNet corrected, jitter NMAR)
    - Per-folder : prior_dir / subfolder / filename[_suffix].raw
                   (body NMAR: aapm_inference_results/body1/file_nmar2.raw)

    Logic: first looks in prior_dir/subfolder/, then falls back to flat prior_dir/.
    """
    stem = art_path.stem
    fname = f"{stem}{prior_suffix}.raw" if prior_suffix else art_path.name

    if subfolder:
        candidate = prior_dir / subfolder / fname
        if candidate.exists():
            return candidate

    return prior_dir / fname


def build_samples(data_root: Path, prior_dir, folder_names: list,
                  prior_suffix: str = '', no_prior: bool = False,
                  angle_dir=None, angle_suffix: str = '') -> list:
    """
    Returns a list of tuples (art_path, prior_path|None, tgt_path, msk_path, angle_path|None).

    no_prior=True  : prior_path=None, U-Net receives 1 channel (art only)
    angle_dir      : second prior as a 3rd channel (e.g. CircNet when prior=NMAR)
    """
    samples = []
    for fname in folder_names:
        root = find_subfolders(data_root / fname)
        baseline_dir = root / 'Baseline'
        target_dir   = root / 'Target'
        mask_dir     = root / 'Mask'

        art_map = {
            int(IDX_PAT.search(f.name).group(1)): f
            for f in sorted(baseline_dir.glob('*img*.raw'))
            if IDX_PAT.search(f.name)
        }
        tgt_map = {
            int(IDX_PAT.search(f.name).group(1)): f
            for f in sorted(target_dir.glob('*img*.raw'))
            if IDX_PAT.search(f.name)
        }
        msk_map = {
            int(re.search(r'(\d+)', f.name).group(1)): f
            for f in sorted(mask_dir.glob('*mask*.raw'))
        }

        for iid in sorted(set(art_map) & set(tgt_map) & set(msk_map)):
            art_path = art_map[iid]

            # Prior (optional) — looks in prior_dir/fname/ then flat prior_dir/
            if no_prior:
                p_path = None
            else:
                p_path = prior_path_for(art_path, prior_dir, prior_suffix, subfolder=fname)
                if not p_path.exists():
                    print(f'[WARN] Missing prior: {p_path} — skipping.')
                    continue

            # Angle-prior (optional 3rd channel)
            if angle_dir is not None:
                a_path = prior_path_for(art_path, angle_dir, angle_suffix, subfolder=fname)
                if not a_path.exists():
                    print(f'[WARN] Missing angle prior: {a_path} — skipping.')
                    continue
            else:
                a_path = None

            samples.append((art_path, p_path, tgt_map[iid], msk_map[iid], a_path))

    return samples


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class CircNetRefinerDataset(Dataset):
    """
    Returns per image:
        x   : [2 or 3, 512, 512]  — (norm. artifact, norm. CircNet-corrected [, seg_channel])
        gt  : [1, 512, 512]       — target: clean outside metal + original inside metal
        m   : [1, 512, 512]       — binary metal mask (float32)
        iid : int                 — image index

    seg_channel_dir : optional directory with seg_channel_img{N}_512x512x1.raw files
                      (generated by graph_channel_generator.py)
    """
    def __init__(self, samples: list, no_prior: bool = False,
                 seg_channel_dir: Path = None,
                 angle_dir: Path = None, angle_suffix: str = ''):
        self.samples         = samples
        self.no_prior        = no_prior
        self.seg_channel_dir = seg_channel_dir
        self.angle_dir       = angle_dir
        self.angle_suffix    = angle_suffix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        art_path, corr_path, tgt_path, msk_path, angle_path = self.samples[idx]

        art  = np.clip(load_raw(art_path),  HU_MIN, HU_MAX)
        tgt  = np.clip(load_raw(tgt_path),  HU_MIN, HU_MAX)
        msk  = load_raw(msk_path)

        art_n = normalize(art)
        tgt_n = normalize(tgt)
        mask  = (msk > 0.5).astype(np.float32)            # [H, W]

        # Target: clean tissue + original in metal region
        gt_n = (1.0 - mask) * tgt_n + mask * art_n        # [H, W]

        channels = [art_n]

        # Prior (channel 1) — skipped when no_prior
        if not self.no_prior and corr_path is not None:
            corr = np.clip(load_raw(corr_path), HU_MIN, HU_MAX)
            channels.append(normalize(corr))

        # Angle-prior (channel 2) — optional second prior
        if angle_path is not None:
            angle = np.clip(load_raw(angle_path), HU_MIN, HU_MAX)
            channels.append(normalize(angle))

        # Semantic channel (segmentation graph) — optional last channel
        if self.seg_channel_dir is not None:
            iid = int(IDX_PAT.search(art_path.name).group(1))
            seg_path = self.seg_channel_dir / f"seg_channel_img{iid}_512x512x1.raw"
            if seg_path.exists():
                seg_ch = load_raw(seg_path).clip(0.0, 1.0)
            else:
                seg_ch = np.zeros_like(art_n, dtype=np.float32)
            channels.append(seg_ch)

        x  = np.stack(channels, axis=0)                   # [1/2/3/4, H, W]
        gt = gt_n[np.newaxis]                              # [1, H, W]
        m  = mask[np.newaxis]                              # [1, H, W]

        iid = int(IDX_PAT.search(art_path.name).group(1))
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(gt).float(),
            torch.from_numpy(m).float(),
            iid,
        )


# -------------------------------------------------------------------------
# Model — U-Net (identical to the one in jitter_refiner_train.py)
# -------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch,   base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base,    base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base*2,  base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base*4,  base * 8)

        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8,  base * 4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4,  base * 2)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.dec1 = ConvBlock(base*2,  base)
        self.out  = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# -------------------------------------------------------------------------
# Forward + Loss
# -------------------------------------------------------------------------
def forward_refine(model: nn.Module, x: torch.Tensor,
                   m: torch.Tensor, hard_paste: bool = True,
                   base_ch: int = 1) -> torch.Tensor:
    """
    x       : [B, C, H, W]  C=1 (no_prior) / 2 (art+prior) / 3+ (art+prior+angle/seg)
    m       : [B, 1, H, W]  metal mask (float32)
    base_ch : base channel for residual prediction
              0 = art (no_prior mode or --base-art), 1 = prior (default)
    Returns : [B, 1, H, W]  prediction in normalised space [0, 1]
    """
    art   = x[:, 0:1]              # [B,1,H,W]
    base  = x[:, base_ch:base_ch+1]  # art or prior
    delta = model(x)               # [B,1,H,W]

    pred = base + (1.0 - m) * delta
    if hard_paste:
        pred = (1.0 - m) * pred + m * art
    return pred


def masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((a - b) * mask))


def loss_fn(pred, gt, art, m, lam_out=1.0, lam_in=0.25):
    """
    lam_out : L1 outside metal vs clean target
    lam_in  : L1 inside metal vs original (stabilisation)
    """
    out_mask = 1.0 - m
    l_out = masked_l1(pred, gt,  out_mask)
    l_in  = masked_l1(pred, art, m)
    total = lam_out * l_out + lam_in * l_in
    return total, {'l_out': l_out.item(), 'l_in': l_in.item()}


# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='CircNet Refiner — U-Net training')
    p.add_argument('--data-root',        type=str, required=True)
    p.add_argument('--prior-dir',        type=str, default=None,
                   help='Directory with prior correction files (CircNet/NMAR). Omit with --no-prior.')
    p.add_argument('--prior-suffix',     type=str, default='',
                   help='Prior filename suffix (CircNet: empty, NMAR: _nmar2)')
    p.add_argument('--no-prior',         action='store_true', default=False,
                   help='No-prior mode — U-Net receives only 1 channel (art)')
    p.add_argument('--base-art',         action='store_true', default=False,
                   help='Residual base = always art (channel 0), prior used as info only')
    p.add_argument('--angle-dir',        type=str, default=None,
                   help='Second prior as an additional channel (e.g. CircNet when prior=NMAR)')
    p.add_argument('--angle-suffix',     type=str, default='',
                   help='Angle-prior filename suffix (e.g. _circnet)')
    p.add_argument('--train-folders',    type=str, nargs='+', required=True)
    p.add_argument('--val-folders',      type=str, nargs='+', required=True)
    p.add_argument('--output-dir',       type=str, required=True)
    p.add_argument('--epochs',           type=int,   default=30)
    p.add_argument('--batch-size',       type=int,   default=4)
    p.add_argument('--num-workers',      type=int,   default=8)
    p.add_argument('--base-features',    type=int,   default=32,
                   help='Base channel dimension of U-Net (32 → ~7.7M params)')
    p.add_argument('--lr',               type=float, default=2e-4)
    p.add_argument('--weight-decay',     type=float, default=1e-4)
    p.add_argument('--lam-out',          type=float, default=1.0)
    p.add_argument('--lam-in',           type=float, default=0.25)
    p.add_argument('--seg-channel-dir',  type=str,   default=None,
                   help='Directory with semantic channels (graph_channel_generator.py). '
                        'When provided, U-Net receives 3 input channels.')
    p.add_argument('--no-hard-paste',    action='store_true', default=False)
    p.add_argument('--device',           type=str,   default='cuda')
    return p.parse_args()



# -------------------------------------------------------------------------
# Training log and plots
# -------------------------------------------------------------------------
def save_training_plot(log_csv: 'Path', output_dir: 'Path') -> None:
    """Generate training_curves.png from training_log.csv."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import csv as _csv

        epochs, tr_loss_vals, va_loss_vals = [], [], []
        with open(log_csv) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                tr_loss_vals.append(float(row['tr_loss']))
                va_loss_vals.append(float(row['va_loss']))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(epochs, tr_loss_vals, label='train', linewidth=1.5)
        axes[0].plot(epochs, va_loss_vals, label='val',   linewidth=1.5)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (L1)')
        axes[0].set_title('Training curves'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # val loss zoomed in (easier to see convergence)
        axes[1].plot(epochs, va_loss_vals, color='tab:orange', linewidth=1.5)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val Loss (L1)')
        axes[1].set_title('Val Loss (detail)'); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / 'training_curves.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Training plot saved: {out_path}')
    except Exception as e:
        print(f'[WARN] Failed to generate training plot: {e}')


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    hard_paste = not args.no_hard_paste

    no_prior        = args.no_prior
    data_root       = Path(args.data_root)
    prior_dir       = Path(args.prior_dir) if args.prior_dir else None
    angle_dir       = Path(args.angle_dir) if args.angle_dir else None
    output_dir      = Path(args.output_dir)
    seg_channel_dir = Path(args.seg_channel_dir) if args.seg_channel_dir else None

    # base_ch: base channel for residual prediction
    # 0 = art (no_prior or --base-art), 1 = prior (default)
    base_ch = 0 if (no_prior or args.base_art) else 1

    output_dir.mkdir(parents=True, exist_ok=True)
    # --- Training log initialisation ---
    log_csv  = output_dir / 'training_log.csv'
    log_file = open(log_csv, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'tr_loss', 'tr_out', 'tr_in', 'va_loss', 'va_out', 'va_in'])

    # Compute number of input channels
    in_channels = 1  # art always
    if not no_prior:
        in_channels += 1   # prior (NMAR or CircNet)
    if angle_dir is not None:
        in_channels += 1   # angle-prior (CircNet when prior=NMAR)
    if seg_channel_dir is not None:
        in_channels += 1   # semantic channel (graph)

    if no_prior:
        print('Prior: none (art-only mode)')
    else:
        prior_label = 'NMAR' if args.prior_suffix else 'CircNet'
        print(f'Prior: {prior_label}  (dir={prior_dir}, suffix="{args.prior_suffix}")')
    if angle_dir:
        print(f'Angle-prior: {angle_dir}')
    if seg_channel_dir:
        print(f'Seg-channel: {seg_channel_dir}')
    print(f'U-Net input channels: {in_channels}')

    # --- Data ---
    build_kwargs = dict(no_prior=no_prior, angle_dir=angle_dir,
                        angle_suffix=args.angle_suffix, prior_suffix=args.prior_suffix)
    train_samples = build_samples(data_root, prior_dir, args.train_folders, **build_kwargs)
    val_samples   = build_samples(data_root, prior_dir, args.val_folders,   **build_kwargs)
    print(f'Training samples  : {len(train_samples)}')
    print(f'Validation samples: {len(val_samples)}')

    train_loader = DataLoader(
        CircNetRefinerDataset(train_samples, no_prior=no_prior,
                              seg_channel_dir=seg_channel_dir,
                              angle_dir=angle_dir, angle_suffix=args.angle_suffix),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        CircNetRefinerDataset(val_samples, no_prior=no_prior,
                              seg_channel_dir=seg_channel_dir,
                              angle_dir=angle_dir, angle_suffix=args.angle_suffix),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # --- Model ---
    model  = UNet(in_ch=in_channels, out_ch=1, base=args.base_features).to(device)
    print(f'U-Net parameters: {sum(p.numel() for p in model.parameters()):,}')

    opt    = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(str(device).startswith('cuda')))

    best_val  = float('inf')
    best_path = output_dir / 'best_refiner.pth'

    for epoch in range(args.epochs):
        # ---- Training ----
        model.train()
        tr_loss = tr_out = tr_in = 0.0
        n_tr = 0

        pbar = tqdm(train_loader, desc=f'Ep {epoch+1:3d}/{args.epochs} [train]')
        for x, gt, m, _ in pbar:
            x  = x.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            m  = m.to(device, non_blocking=True)
            art = x[:, 0:1]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(str(device).startswith('cuda'))):
                pred = forward_refine(model, x, m, hard_paste=hard_paste, base_ch=base_ch)
                loss, parts = loss_fn(pred, gt, art, m, args.lam_out, args.lam_in)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs       = x.size(0)
            tr_loss += loss.item() * bs
            tr_out  += parts['l_out'] * bs
            tr_in   += parts['l_in']  * bs
            n_tr    += bs
            pbar.set_postfix(loss=f'{tr_loss/n_tr:.5f}',
                             l_out=f'{tr_out/n_tr:.5f}',
                             l_in=f'{tr_in/n_tr:.5f}')

        tr_loss /= n_tr; tr_out /= n_tr; tr_in /= n_tr

        # ---- Validation ----
        model.eval()
        va_loss = va_out = va_in = 0.0
        n_va = 0

        with torch.no_grad():
            for x, gt, m, _ in tqdm(val_loader, desc=f'Ep {epoch+1:3d}/{args.epochs} [val] '):
                x  = x.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                m  = m.to(device, non_blocking=True)
                art = x[:, 0:1]
                with torch.cuda.amp.autocast(enabled=(str(device).startswith('cuda'))):
                    pred = forward_refine(model, x, m, hard_paste=hard_paste, base_ch=base_ch)
                    loss, parts = loss_fn(pred, gt, art, m, args.lam_out, args.lam_in)
                bs       = x.size(0)
                va_loss += loss.item() * bs
                va_out  += parts['l_out'] * bs
                va_in   += parts['l_in']  * bs
                n_va    += bs

        va_loss /= max(n_va, 1); va_out /= max(n_va, 1); va_in /= max(n_va, 1)

        print(f'[Ep {epoch+1:3d}] train={tr_loss:.6f} '
              f'(out={tr_out:.6f}, in={tr_in:.6f}) | '
              f'val={va_loss:.6f} (out={va_out:.6f}, in={va_in:.6f})')
        log_writer.writerow([epoch+1, f'{tr_loss:.6f}', f'{tr_out:.6f}', f'{tr_in:.6f}',
                              f'{va_loss:.6f}', f'{va_out:.6f}', f'{va_in:.6f}'])
        log_file.flush()

        ckpt = {
            'epoch':    epoch + 1,
            'model':    model.state_dict(),
            'opt':      opt.state_dict(),
            'val_loss': va_loss,
            'args':     vars(args),
        }
        torch.save(ckpt, output_dir / f'refiner_epoch{epoch+1:03d}.pth')

        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, best_path)
            print(f'  → New best: {best_path}')

    log_file.close()
    save_training_plot(log_csv, output_dir)
    print(f'\nTraining complete. Best val_loss: {best_val:.6f}')
    print(f'Best model: {best_path}')


if __name__ == '__main__':
    main()
