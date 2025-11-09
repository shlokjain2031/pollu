#!/usr/bin/env python3
"""Validate whether a GeoTIFF is likely a DSM (Digital Surface Model).

Produces a short human report and exits 0. Uses heuristic checks:
- metadata keywords (DSM, SURFACE, COPERNICUS)
- global statistics (min/max/percentiles)
- gradient magnitude (downsampled) to detect rough surface
- local window variability sampled at random points to detect building/vegetation signatures

Usage: scripts/validate_dsm.py path/to/output_hh.tif
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

try:
    import numpy as np
    import rasterio
    from rasterio.windows import Window
except Exception as e:
    print('Missing required Python packages. Please install dependencies:')
    print('  pip install numpy rasterio')
    print('Error detail:', e)
    raise


def read_overview_stats(src, max_dim=1024):
    # read a decimated overview for quick stats and gradients
    h = src.height
    w = src.width
    scale = max(1, max(h // max_dim, w // max_dim))
    out_h = h // scale
    out_w = w // scale
    arr = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
    return arr, scale


def pct(arr, q):
    return float(np.nanpercentile(arr, q))


def sample_local_stats(src, n_samples=100, window_size=11, seed=2):
    rng = random.Random(seed)
    h = src.height
    w = src.width
    pad = window_size // 2
    stats = []
    for _ in range(n_samples):
        # sample a center pixel away from the border
        y = rng.randint(pad, max(pad, h - pad - 1))
        x = rng.randint(pad, max(pad, w - pad - 1))
        win = Window(x - pad, y - pad, window_size, window_size)
        try:
            data = src.read(1, window=win)
        except Exception:
            continue
        if data.size == 0:
            continue
        data = data.astype('float32')
        # mask nodata
        if src.nodata is not None:
            data = data[data != src.nodata]
        if data.size == 0:
            continue
        stats.append({
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.max(data) - np.min(data)),
        })
    return stats


def heuristic_dsm_decision(meta, arr, grad_mag, local_stats):
    decisions = []
    # metadata hints
    tags = meta.get('tags', {})
    tag_text = ' '.join([str(v).upper() for v in tags.values()]) if tags else ''
    if any(k in tag_text for k in ('DSM', 'SURFACE', 'COPERNICUS', 'DSM_COG', 'DSM30')):
        decisions.append('metadata: contains DSM-like keywords')

    # global stats
    minv = float(np.nanmin(arr))
    maxv = float(np.nanmax(arr))
    rng = maxv - minv
    if rng > 10.0:
        decisions.append(f'global range {rng:.1f} m (>10 m) suggests surface variation')

    # gradient magnitude
    gm_mean = float(np.nanmean(grad_mag))
    gm_std = float(np.nanstd(grad_mag))
    if gm_mean > 0.1:
        decisions.append(f'gradient mean {gm_mean:.3f} (non-smooth surface)')

    # local stats: fraction of windows with non-trivial range
    ranges = [s['range'] for s in local_stats if 'range' in s]
    frac_high = float(sum(1 for r in ranges if r > 2.0) / max(1, len(ranges)))
    if frac_high > 0.25:
        decisions.append(f'{frac_high:.2f} fraction of local windows have range>2m (buildings/trees likely)')

    # final heuristic
    likely = False
    if decisions:
        likely = True
    else:
        # if none of the heuristics triggered, still check if global range tiny -> likely DTM
        if rng < 3.0 and gm_mean < 0.05 and frac_high < 0.05:
            likely = False
        else:
            likely = True

    return likely, decisions


def make_report(path: Path, args) -> int:
    if not path.exists():
        print(f'ERROR: file not found: {path}')
        return 2

    with rasterio.open(path) as src:
        meta = {
            'crs': str(src.crs),
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': str(src.dtypes[0]),
            'transform': tuple(src.transform),
            'nodata': src.nodata,
            'tags': src.tags(),
        }

        print('File:', path)
        print('CRS:', meta['crs'])
        print('Size (px):', meta['width'], 'x', meta['height'])
        print('Dtype:', meta['dtype'], 'bands:', meta['count'])
        print('nodata:', meta['nodata'])

        # quick overview
        arr, scale = read_overview_stats(src, max_dim=1024)
        arr = arr.astype('float32')
        if meta['nodata'] is not None:
            arr[arr == meta['nodata']] = np.nan

        print('\nOverview stats (downsampled):')
        print(' min:', float(np.nanmin(arr)), ' max:', float(np.nanmax(arr)))
        for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
            print(f' p{q}:', pct(arr, q))

        # gradient magnitude on overview
        gy, gx = np.gradient(np.nan_to_num(arr, nan=np.nanmean(arr)))
        grad_mag = np.hypot(gx, gy)
        print('\nGradient stats (overview): mean {:.4f}, std {:.4f}'.format(float(np.nanmean(grad_mag)), float(np.nanstd(grad_mag))))

        # sample local windows at full res
        print('\nSampling local windows (full resolution)...')
        local_stats = sample_local_stats(src, n_samples=args.samples, window_size=args.window)
        if local_stats:
            ls_arr = np.array([s['range'] for s in local_stats])
            print(' local windows sampled:', len(local_stats))
            print(' local range: mean {:.2f} m, median {:.2f} m, p90 {:.2f} m'.format(float(np.mean(ls_arr)), float(np.median(ls_arr)), float(np.nanpercentile(ls_arr, 90))))
        else:
            print(' no valid local samples (all nodata?)')

        likely, reasons = heuristic_dsm_decision(meta, arr, grad_mag, local_stats)

        print('\nHeuristic decision:')
        if likely:
            print(' -> LIKELY DSM (surface model)')
        else:
            print(' -> LIKELY NOT DSM (probably DTM or very smooth terrain)')

        if reasons:
            print('\nReasons:')
            for r in reasons:
                print(' -', r)

        # write JSON summary if requested
        if args.json_out:
            out = {
                'path': str(path),
                'meta': meta,
                'overview': {
                    'min': float(np.nanmin(arr)),
                    'max': float(np.nanmax(arr)),
                    'p50': float(pct(arr, 50)),
                },
                'gradient_mean': float(np.nanmean(grad_mag)),
                'local_samples': local_stats,
                'likely_dsm': bool(likely),
                'reasons': reasons,
            }
            with open(args.json_out, 'w') as fh:
                json.dump(out, fh, indent=2)
            print('\nWrote JSON summary to', args.json_out)

        return 0


def main():
    parser = argparse.ArgumentParser(description='Simple DSM validator heuristics')
    parser.add_argument('tif', help='path to GeoTIFF to validate')
    parser.add_argument('--samples', type=int, default=80, help='number of local windows to sample (default 80)')
    parser.add_argument('--window', type=int, default=11, help='window size in pixels for local stats (odd number, default 11)')
    parser.add_argument('--json-out', help='write JSON summary to path')
    args = parser.parse_args()

    rc = make_report(Path(args.tif), args)
    sys.exit(rc)


if __name__ == '__main__':
    main()
