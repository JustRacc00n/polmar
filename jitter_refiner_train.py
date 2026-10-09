"""
jitter_refiner_train.py — Refiner (U-Net) training on the Jitter dataset

Dataset: clean hospital CT scans with synthetically added metal artifacts.
  - Artifact images:  jitter_Clean{NN}_slice{NNNN}_H512_W512.raw
  - Clean images:     Clean{NN}_slice{NNNN}_H512_W512.raw
  - NMAR prior:       jitter_Clean{NN}_slice{NNNN}_H512_W512_nmar2.raw
  - SEG channel:      jitter_Clean{NN}_slice{NNNN}_H512_W512_seg_channel.raw  (optional)
  - Metal mask:       derived from the artifact image (HU > --metal-hu)

Network pipeline:
  Input:      [art, nmar]           — 2 channels (Base+NMAR)
          or  [art, nmar, seg_chan] — 3 channels with semantic SEG channel
  Prediction: nmar + (1 − mask) × delta
  Hard paste: metal pixels in output = original artifact (unchanged)
  Target:     clean outside metal + original inside metal

Train/val split: per scan_id.
  --train-scan-ids 1 2 3 4 5 6 7
  --val-scan-ids   8 9

Example:
    python3 jitter_refiner_train.py \
        --art-dir       /app/data/3frames_jitter_raw_clean_real \
        --clean-dir     /app/data/Clean_raw \
        --nmar-dir      /app/output/nmar_results \
        --seg-chan-dir  /app/data/seg_channels_jitter \
        --train-scan-ids 1 2 3 4 5 6 7 \
        --val-scan-ids   8 9 \
        --output-dir    /app/CircNet/refiner_jitter_run_${SLURM_JOB_ID} \
        --epochs 50 --batch-size 4
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

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
HU_MIN, HU_MAX   = -1000.0, 1000.0   # HU window for normalisation
METAL_HU_DEFAULT = 2500.0             # HU threshold for metal mask

IMG_SIZE = 512

ART_RE     = re.compile(r"jitter_Clean(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+\.raw$")
CLEAN_RE   = re.compile(r"Clean(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+\.raw$")
SEG_CH_RE  = re.compile(r"jitter_Clean(?P<scan>\d+)_slice(?P<slice>\d+)_H\d+_W\d+_seg_channel\.raw$")


# -------------------------------------------------------------------------
# Normalisation
# -------------------------------------------------------------------------
def normalize(x: np.ndarray) -> np.ndarray:
    """Clip to [HU_MIN, HU_MAX] and map to [0, 1]."""
    return (np.clip(x, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)


def load_raw(path) -> np.ndarray:
    """Load a float32 raw file of shape [512×512]."""
    return np.fromfile(str(path), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)


# -------------------------------------------------------------------------
# Sample indexing
# -------------------------------------------------------------------------
def _index_dir(directory: Path, regex: re.Pattern, key_groups=("scan", "slice")):
    """Return dict {(scan_id, slice_id): Path} for all matching files in directory."""
    result = {}
    for p in sorted(directory.glob("*.raw")):
        m = regex.search(p.name)
        if m:
            key = tuple(int(m.group(g)) for g in key_groups)
            result[key] = p
    return result


def build_samples_jitter(
    art_dir: Path, clean_dir: Path, prior_dir,
    seg_chan_dir,
    scan_ids: list,
    prior_suffix: str = '_nmar2',
    no_prior: bool = False,
    angle_dir=None, angle_suffix: str = '',
    use_3frames: bool = False,
) -> list:
    """
    Returns a list of tuples:
      (art_path, clean_path, prior_path|None, seg_path|None, angle_path|None,
       art_prev_path|None, art_next_path|None)

    Last two elements are None when use_3frames=False.

    no_prior=True  : prior_path=None, U-Net receives 1 channel (art only)
    angle_dir      : second prior as an additional channel (e.g. CircNet when prior=NMAR)
    use_3frames    : adds neighbouring slices art_prev/art_next (fallback: current slice)
    """
    art_map   = _index_dir(art_dir,   ART_RE)
    clean_map = _index_dir(clean_dir, CLEAN_RE)

    seg_map = {}
    if seg_chan_dir is not None:
        seg_map = _index_dir(seg_chan_dir, SEG_CH_RE)

    scan_set = set(scan_ids)
    samples = []
    missing_prior = missing_clean = missing_angle = 0

    for (scan_id, slice_id), art_p in sorted(art_map.items()):
        if scan_id not in scan_set:
            continue
        key = (scan_id, slice_id)
        if key not in clean_map:
            missing_clean += 1
            continue

        # Prior
        if no_prior:
            prior_p = None
        else:
            prior_p = prior_dir / f'{art_p.stem}{prior_suffix}.raw'
            if not prior_p.exists():
                missing_prior += 1
                continue

        # Angle-prior (optional 3rd channel, e.g. CircNet output)
        if angle_dir is not None:
            angle_p = angle_dir / f'{art_p.stem}{angle_suffix}.raw'
            if not angle_p.exists():
                missing_angle += 1
                continue
        else:
            angle_p = None

        seg_p = seg_map.get(key, None)

        # Neighbouring slices — art or prior depending on variant
        if use_3frames:
            if no_prior:
                # 3f-base: neighbours from art_dir → [art_prev, art_curr, art_next]
                frame_prev_p = art_map.get((scan_id, slice_id - 1), art_p)
                frame_next_p = art_map.get((scan_id, slice_id + 1), art_p)
            else:
                # 3f-circnet: neighbours from prior_dir → [circ_prev, circ_curr, circ_next]
                art_prev_p = art_map.get((scan_id, slice_id - 1), art_p)
                art_next_p = art_map.get((scan_id, slice_id + 1), art_p)
                frame_prev_p = (prior_dir / f'{art_prev_p.stem}{prior_suffix}.raw'
                                if art_prev_p.exists() else prior_p)
                frame_next_p = (prior_dir / f'{art_next_p.stem}{prior_suffix}.raw'
                                if art_next_p.exists() else prior_p)
                if not frame_prev_p.exists():
                    frame_prev_p = prior_p
                if not frame_next_p.exists():
                    frame_next_p = prior_p
        else:
            frame_prev_p = None
            frame_next_p = None

        samples.append((art_p, clean_map[key], prior_p, seg_p, angle_p,
                         frame_prev_p, frame_next_p))

    if missing_clean:
        print(f"  [WARN] Missing clean for {missing_clean} images — skipped.")
    if missing_prior:
        print(f"  [WARN] Missing prior (suffix '{prior_suffix}') for {missing_prior} images — skipped.")
    if missing_angle:
        print(f"  [WARN] Missing angle prior (suffix '{angle_suffix}') for {missing_angle} images — skipped.")

    return samples


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class JitterRefinerDataset(Dataset):
    """
    Returns per image:
        x   : [C, 512, 512]  — input channels (see channel order below)
        gt  : [1, 512, 512]  — target: clean outside metal + artifact inside metal
        m   : [1, 512, 512]  — binary metal mask (float32)
        iid : (scan_id, slice_id)

    Channel order:
        art_curr [, prior] [, angle] [, seg] [, art_prev, art_next]

    art_curr is always at index 0 — required by forward_refine.
    Neighbouring slices (art_prev, art_next) appended last when use_3frames=True.
    Metal mask is derived from art_curr (HU > metal_hu).
    """
    def __init__(self, samples: list, metal_hu: float = METAL_HU_DEFAULT,
                 use_seg_channel: bool = False, no_prior: bool = False,
                 use_angle: bool = False, use_3frames: bool = False):
        self.samples         = samples
        self.metal_hu        = metal_hu
        self.use_seg_channel = use_seg_channel
        self.no_prior        = no_prior
        self.use_angle       = use_angle
        self.use_3frames     = use_3frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        art_p, clean_p, prior_p, seg_p, angle_p = sample[:5]
        art_prev_p = sample[5] if len(sample) > 5 else None
        art_next_p = sample[6] if len(sample) > 6 else None

        art   = load_raw(art_p)
        clean = load_raw(clean_p)

        # Metal mask: derived from the artifact image (HU > threshold)
        mask = (art > self.metal_hu).astype(np.float32)       # [H, W]

        art_n   = normalize(art)
        clean_n = normalize(clean)

        # Target: clean outside metal, original artifact inside metal
        gt_n = (1.0 - mask) * clean_n + mask * art_n          # [H, W]

        if self.use_3frames:
            # 3-frame variants — custom channel order
            frame_prev = load_raw(art_prev_p) if art_prev_p is not None else art
            frame_next = load_raw(art_next_p) if art_next_p is not None else art
            if self.no_prior:
                # 3f-base: [art_prev, art_curr, art_next]
                channels = [normalize(frame_prev), art_n, normalize(frame_next)]
            else:
                # 3f-circnet: [circ_prev, circ_curr, circ_next]
                prior_n = normalize(load_raw(prior_p))
                channels = [normalize(frame_prev), prior_n, normalize(frame_next)]
        else:
            # Standard variants — art_curr at index 0
            channels = [art_n]

            if not self.no_prior and prior_p is not None:
                channels.append(normalize(load_raw(prior_p)))

            if self.use_angle and angle_p is not None:
                channels.append(normalize(load_raw(angle_p)))

            if self.use_seg_channel:
                if seg_p is not None and seg_p.exists():
                    seg_ch = load_raw(seg_p).clip(0.0, 1.0)
                else:
                    seg_ch = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                channels.append(seg_ch)

        x  = np.stack(channels, axis=0)                        # [C, H, W]
        gt = gt_n[np.newaxis]                                  # [1, H, W]
        m  = mask[np.newaxis]                                  # [1, H, W]

        # Parse scan/slice IDs from filename
        mt = ART_RE.search(art_p.name)
        iid = (int(mt.group("scan")), int(mt.group("slice"))) if mt else (0, idx)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(gt).float(),
            torch.from_numpy(m).float(),
            iid,
        )


# -------------------------------------------------------------------------
# Refiner model — U-Net (identical to the one in circnet_refiner_train.py)
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
        self.enc1  = ConvBlock(in_ch,   base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2  = ConvBlock(base,    base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3  = ConvBlock(base*2,  base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4  = ConvBlock(base*4,  base * 8)

        self.up3   = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3  = ConvBlock(base*8,  base * 4)
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2  = ConvBlock(base*4,  base * 2)
        self.up1   = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.dec1  = ConvBlock(base*2,  base)
        self.out   = nn.Conv2d(base, out_ch, 1)

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
                   base_ch: int = 1, art_ch: int = 0) -> torch.Tensor:
    """
    x       : [B, C, H, W]  input channels
    m       : [B, 1, H, W]  metal mask
    base_ch : index of the base channel — delta is added to this channel
    art_ch  : index of the 'art' channel used for hard-paste and l_in loss
              (default 0 = art_curr for standard variants;
               1 = middle slice for 3-frame variants: [prev, curr, next])
    Returns : [B, 1, H, W]  prediction normalised to [0, 1]
    """
    art   = x[:, art_ch:art_ch+1]   # [B, 1, H, W]
    base  = x[:, base_ch:base_ch+1] # [B, 1, H, W]
    delta = model(x)                 # [B, 1, H, W]

    pred = base + (1.0 - m) * delta
    if hard_paste:
        pred = (1.0 - m) * pred + m * art
    return pred


def masked_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((a - b) * mask))


def loss_fn(pred, gt, art, m, lam_out=1.0, lam_in=0.25):
    out_mask = 1.0 - m
    l_out = masked_l1(pred, gt,  out_mask)
    l_in  = masked_l1(pred, art, m)
    total = lam_out * l_out + lam_in * l_in
    return total, {'l_out': l_out.item(), 'l_in': l_in.item()}


# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Jitter Refiner — U-Net training (NMAR prior)')
    # --- Data ---
    p.add_argument('--art-dir',       type=str, required=True,
                   help='Directory with jitter_Clean*_slice*_H512_W512.raw files')
    p.add_argument('--clean-dir',     type=str, required=True,
                   help='Directory with Clean*_slice*_H512_W512.raw files')
    p.add_argument('--prior-dir',     type=str, default=None,
                   help='Directory with prior files. Omit when using --no-prior.')
    p.add_argument('--prior-suffix',  type=str, default='_nmar2',
                   help='Prior file suffix (default _nmar2; for CircNet: _circnet)')
    p.add_argument('--no-prior',      action='store_true', default=False,
                   help='No-prior mode — U-Net receives only 1 channel (art)')
    p.add_argument('--angle-dir',     type=str, default=None,
                   help='Second prior as an additional channel (e.g. CircNet when prior=NMAR)')
    p.add_argument('--angle-suffix',  type=str, default='_circnet',
                   help='Angle-prior file suffix (default _circnet)')
    p.add_argument('--seg-chan-dir',  type=str, default=None,
                   help='Directory with semantic channels (*_seg_channel.raw).')
    p.add_argument('--use-3frames',  action='store_true', default=False,
                   help='Add neighbouring slices art_prev and art_next as extra input channels.')
    # --- Split ---
    p.add_argument('--train-scan-ids', type=int, nargs='+', required=True,
                   help='Scan IDs for training (e.g. 1 2 3 4 5 6 7)')
    p.add_argument('--val-scan-ids',   type=int, nargs='+', required=True,
                   help='Scan IDs for validation (e.g. 8 9)')
    # --- Output ---
    p.add_argument('--output-dir',     type=str, required=True)
    # --- Training ---
    p.add_argument('--epochs',         type=int,   default=50)
    p.add_argument('--batch-size',     type=int,   default=4)
    p.add_argument('--num-workers',    type=int,   default=8)
    p.add_argument('--base-features',  type=int,   default=32)
    p.add_argument('--lr',             type=float, default=2e-4)
    p.add_argument('--weight-decay',   type=float, default=1e-4)
    p.add_argument('--lam-out',        type=float, default=1.0)
    p.add_argument('--lam-in',         type=float, default=0.25)
    p.add_argument('--metal-hu',       type=float, default=METAL_HU_DEFAULT,
                   help='HU threshold for metal mask derivation (default 2500)')
    p.add_argument('--no-hard-paste',  action='store_true', default=False)
    p.add_argument('--device',         type=str,   default='cuda')
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

    art_dir      = Path(args.art_dir)
    clean_dir    = Path(args.clean_dir)
    prior_dir    = Path(args.prior_dir) if args.prior_dir else None
    seg_chan_dir  = Path(args.seg_chan_dir) if args.seg_chan_dir else None
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- Training log initialisation ---
    log_csv  = output_dir / 'training_log.csv'
    log_file = open(log_csv, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'tr_loss', 'tr_out', 'tr_in', 'va_loss', 'va_out', 'va_in'])

    no_prior     = args.no_prior
    angle_dir    = Path(args.angle_dir) if args.angle_dir else None
    if not no_prior and prior_dir is None:
        raise ValueError('Either --prior-dir or --no-prior is required.')

    use_seg      = seg_chan_dir is not None
    use_angle    = angle_dir is not None
    use_3frames  = args.use_3frames

    if use_3frames:
        # [prev, curr, next] — always 3 channels, curr at index 1
        in_channels = 3
        base_ch     = 1   # middle slice (art_curr or circ_curr)
        art_ch      = 1   # same slice for hard-paste and l_in
        ch_names    = (['art_prev', 'art_curr', 'art_next'] if no_prior
                       else ['circ_prev', 'circ_curr', 'circ_next'])
    else:
        in_channels  = 1                           # art_curr
        in_channels += 0 if no_prior else 1        # prior
        in_channels += 1 if use_angle  else 0      # angle
        in_channels += 1 if use_seg    else 0      # seg
        base_ch      = 0 if no_prior else 1
        art_ch       = 0                           # art_curr always at index 0
        ch_names     = ['art_curr']
        if not no_prior: ch_names.append('prior')
        if use_angle:    ch_names.append('angle')
        if use_seg:      ch_names.append('seg')

    prior_label = 'none' if no_prior else ('CircNet' if args.prior_suffix == '_circnet' else f'prior({args.prior_suffix})')
    print(f'Prior: {prior_label}{"  angle: " + str(angle_dir) if use_angle else ""}')
    print(f'3-frames: {use_3frames}')
    print(f'U-Net input channels: {in_channels}  {ch_names}  (base_ch={base_ch}  art_ch={art_ch})')
    print(f'Metal HU threshold: {args.metal_hu}')

    # --- Dane ---
    build_kwargs = dict(prior_suffix=args.prior_suffix, no_prior=no_prior,
                        angle_dir=angle_dir, angle_suffix=args.angle_suffix,
                        use_3frames=use_3frames)
    train_samples = build_samples_jitter(
        art_dir, clean_dir, prior_dir, seg_chan_dir, args.train_scan_ids, **build_kwargs)
    val_samples   = build_samples_jitter(
        art_dir, clean_dir, prior_dir, seg_chan_dir, args.val_scan_ids,   **build_kwargs)

    print(f'Training scans:    {sorted(args.train_scan_ids)}  → {len(train_samples)} samples')
    print(f'Validation scans:  {sorted(args.val_scan_ids)}   → {len(val_samples)} samples')

    if not train_samples:
        raise RuntimeError("No training samples found! Check paths and scan_ids.")

    # Check how many samples have a seg_channel
    if use_seg:
        n_with_seg = sum(1 for s in train_samples if s[3] is not None)
        print(f'  (train) with semantic channel: {n_with_seg}/{len(train_samples)}')

    ds_kwargs = dict(metal_hu=args.metal_hu, use_seg_channel=use_seg,
                     no_prior=no_prior, use_angle=use_angle, use_3frames=use_3frames)
    train_loader = DataLoader(
        JitterRefinerDataset(train_samples, **ds_kwargs),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        JitterRefinerDataset(val_samples, **ds_kwargs),
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
            art = x[:, art_ch:art_ch+1]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(str(device).startswith('cuda'))):
                pred = forward_refine(model, x, m, hard_paste=hard_paste,
                                      base_ch=base_ch, art_ch=art_ch)
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
                art = x[:, art_ch:art_ch+1]
                with torch.cuda.amp.autocast(enabled=(str(device).startswith('cuda'))):
                    pred = forward_refine(model, x, m, hard_paste=hard_paste,
                                          base_ch=base_ch, art_ch=art_ch)
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
            'art_ch':   art_ch,
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
