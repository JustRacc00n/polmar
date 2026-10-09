"""
collect_metrics.py — Zbiorcza tabela metryk ze wszystkich runów refinera

Przeszukuje katalogi runów (wg wzorca nazwy lub podanej listy ścieżek),
odczytuje plik metrics/summary.json z każdego runu i agreguje wyniki
do jednej tabeli CSV oraz wydruku na konsoli.

Użycie:
  # Wszystkie runy z katalogu roboczego CircNet
  python3 collect_metrics.py \
      --runs-dir /app/CircNet \
      --output   /app/CircNet/all_metrics_summary.csv

  # Konkretne katalogi
  python3 collect_metrics.py \
      --run-dirs refiner_body_nmar_64f_12345 refiner_body_angle_64f_67890 \
      --output   /app/CircNet/all_metrics_summary.csv

  # Podgląd bez zapisu
  python3 collect_metrics.py --runs-dir /app/CircNet --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


METRICS_DISPLAY = [
    ('MAE_art  [HU]',   'mae_art',    'mean'),
    ('MAE_prior[HU]',   'mae_prior',  'mean'),
    ('MAE_prior2[HU]',  'mae_angle',  'mean'),   # 2. prior (np. CircNet gdy główny=NMAR)
    ('MAE_pred [HU]',   'mae_pred',   'mean'),
    ('RMSE_art [HU]',   'rmse_art',   'mean'),
    ('RMSE_pred[HU]',   'rmse_pred',  'mean'),
    ('PSNR_art [dB]',   'psnr_art',   'mean'),
    ('PSNR_prior[dB]',  'psnr_prior', 'mean'),
    ('PSNR_prior2[dB]', 'psnr_angle', 'mean'),
    ('PSNR_pred[dB]',   'psnr_pred',  'mean'),
    ('SSIM_art',        'ssim_art',   'mean'),
    ('SSIM_prior',      'ssim_prior', 'mean'),
    ('SSIM_prior2',     'ssim_angle', 'mean'),
    ('SSIM_pred',       'ssim_pred',  'mean'),
    ('GradMAE_pred',    'grad_pred',  'mean'),
]


def parse_args():
    p = argparse.ArgumentParser(description='Zbiorcza tabela metryk refinera')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--runs-dir', type=str,
                     help='Katalog zawierający foldery runów (szuka rekurencyjnie summary.json)')
    src.add_argument('--run-dirs', type=str, nargs='+',
                     help='Lista konkretnych katalogów runów')
    p.add_argument('--output',  type=str, default=None,
                   help='Ścieżka wyjściowego CSV (domyślnie: runs_dir/all_metrics_summary.csv)')
    p.add_argument('--dry-run', action='store_true',
                   help='Tylko wydruk na konsoli, bez zapisu CSV')
    p.add_argument('--sort-by', type=str, default='psnr_pred_mean',
                   choices=['psnr_pred_mean', 'ssim_pred_mean', 'mae_pred_mean'],
                   help='Kolumna do sortowania (domyślnie: psnr_pred_mean)')
    return p.parse_args()


def find_summary_files(runs_dir: Path) -> list:
    """Zwraca posortowaną listę (run_name, summary_path)."""
    results = []
    for summary_path in sorted(runs_dir.glob('*/metrics/summary.json')):
        run_name = summary_path.parent.parent.name
        results.append((run_name, summary_path))
    return results


def load_summary(run_name: str, summary_path: Path) -> dict:
    with open(summary_path) as f:
        data = json.load(f)

    row = {'run': run_name}

    # Metadane
    for k in ('test_folder', 'test_scan_id', 'n_images', 'prior', 'prior_suffix'):
        if k in data:
            row[k] = data[k]

    # Metryki — spłaszcz {metric: {mean, std}} → metric_mean, metric_std
    for metric, vals in data.get('metrics', {}).items():
        if isinstance(vals, dict):
            row[f'{metric}_mean'] = vals.get('mean', float('nan'))
            row[f'{metric}_std']  = vals.get('std',  float('nan'))
        else:
            row[f'{metric}_mean'] = float(vals)

    return row


def print_table(df: pd.DataFrame) -> None:
    """Czytelny wydruk najważniejszych metryk."""
    col_w  = 18
    suf_w  = 10   # szerokość kolumny "prior"

    # Całkowita szerokość = 40 (run) + suf_w+2 (prior) + N*(col_w)
    total_w = 40 + suf_w + 2 + len(METRICS_DISPLAY) * col_w
    print(f'\n{"="*total_w}')
    print(f'  Zbiorcze metryki refinera  ({len(df)} runów)')
    print(f'{"="*total_w}')

    header = f'{"Run":<40}  {"Prior":<{suf_w}}'
    for label, _, _ in METRICS_DISPLAY:
        header += f'  {label:>{col_w-2}}'
    print(header)
    print('-' * len(header))

    for _, row in df.iterrows():
        # Kolumna prior: pokaż suffix jeśli jest, inaczej "—" (no_prior) lub "?"
        suf = str(row.get('prior_suffix', row.get('prior', '?')))
        line = f'{str(row["run"]):<40}  {suf:<{suf_w}}'
        for _, metric, stat in METRICS_DISPLAY:
            col = f'{metric}_{stat}'
            val = row.get(col, float('nan'))
            if pd.isna(val):
                line += f'  {"—":>{col_w-2}}'
            else:
                line += f'  {val:>{col_w-2}.4f}'
        print(line)

    print(f'\n{"="*total_w}')
    print('Legenda:')
    print('  prior   = typ/suffix głównego prioru (np. _nmar2, _circnet, lub "—" gdy brak)')
    print('  prior2  = drugi prior (CircNet gdy główny=NMAR) — wypełniony tylko gdy wariant ma 2 priory')
    print('  art=artefakt (wejście)  prior=1.prior  prior2=2.prior  pred=wyjście refinera')


def main():
    args = parse_args()

    # Zbierz summary.json
    summaries = []
    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
        summaries = find_summary_files(runs_dir)
        if not summaries:
            print(f'[ERROR] Nie znaleziono żadnych plików metrics/summary.json w: {runs_dir}')
            print('        Czy runy są ukończone i mają wygenerowane metryki?')
            sys.exit(1)
        out_default = runs_dir / 'all_metrics_summary.csv'
    else:
        for d in args.run_dirs:
            p = Path(d)
            sp = p / 'metrics' / 'summary.json'
            if sp.exists():
                summaries.append((p.name, sp))
            else:
                print(f'[WARN] Brak {sp} — pomijam.')
        if not summaries:
            print('[ERROR] Żaden z podanych katalogów nie ma metrics/summary.json.')
            sys.exit(1)
        out_default = Path(args.run_dirs[0]).parent / 'all_metrics_summary.csv'

    print(f'Znaleziono {len(summaries)} runów z metrykami.')

    rows = []
    for run_name, sp in summaries:
        try:
            rows.append(load_summary(run_name, sp))
        except Exception as e:
            print(f'[WARN] Błąd przy wczytaniu {sp}: {e}')

    if not rows:
        print('[ERROR] Żadne metryki nie zostały wczytane.')
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Sortowanie
    sort_col = args.sort_by
    ascending = sort_col.startswith('mae')  # MAE niższe = lepsze; PSNR/SSIM wyższe = lepsze
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending)

    print_table(df)

    if not args.dry_run:
        out_path = Path(args.output) if args.output else out_default
        df.to_csv(out_path, index=False, float_format='%.6f')
        print(f'\nTabela zapisana: {out_path}')
    else:
        print('\n[dry-run] Pominięto zapis CSV.')


if __name__ == '__main__':
    main()
