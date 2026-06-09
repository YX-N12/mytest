"""
main.py  –  Full DCT-SVD watermarking experiment pipeline.

Run from the project directory:
    python main.py

Output directory tree (created automatically under ./output/):
    output/
    ├── watermarked/
    │   ├── cover_vs_watermarked.png       — imperceptibility check
    │   └── singular_value_profiles.png    — SV magnitude before/after
    ├── attacks/
    │   └── <attack_name>.png              — one attacked image per attack
    ├── extracted/
    │   ├── no_attack/grid.png             — 2×2 per-quadrant watermarks
    │   └── <attack_name>/grid.png
    ├── plots/
    │   ├── intensity_curves_all.png       — 3×4 robustness-vs-strength figure
    │   ├── intensity_<attack>.png         — individual intensity curves
    │   ├── baseline_comparison.png        — DCT-SVD vs baseline (best quad)
    │   ├── baseline_comparison_all.png    — per-quadrant full comparison
    │   ├── heatmap_sv_corr.png            — SV correlation heatmap
    │   └── heatmap_pixel_corr.png         — pixel correlation heatmap
    └── tables/
        ├── sv_correlation_table.csv
        ├── pixel_correlation_table.csv
        ├── baseline_pixel_corr.csv
        ├── psnr_ssim.csv
        └── intensity_results.csv
"""

import sys
import time
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# ── local modules ─────────────────────────────────────────────────────────────
from config import (
    COVER_PATH, WATERMARK_PATH, DIRS, OUTPUT_DIR,
    QUAD_NAMES, ATTACK_DEFAULTS, ATTACK_LEVELS, ATTACK_DISPLAY,
    QUAD_SHAPE, WM_SHAPE,
)
from utils                import normalize_display
from watermark_algorithm  import load_images, embed_watermark, extract_watermark
from baseline_dct         import embed_baseline, extract_baseline
from attacks              import apply_attack
from metrics              import psnr, ssim_metric
from visualization        import (
    save_cover_vs_watermarked,
    save_sv_profiles,
    save_extracted_grid,
    save_intensity_curves_all,
    save_intensity_curves_single,
    save_baseline_comparison,
    save_baseline_comparison_all_quads,
    save_correlation_heatmap,
    save_attack_gallery,
    save_all_attacks_grid,
    save_all_baseline_attacks_grid,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_dirs() -> None:
    """Create all output sub-directories."""
    for d in DIRS.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    (DIRS['baseline'] / 'extracted').mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]  {msg}", flush=True)


def save_image(img: np.ndarray, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    make_dirs()
    np.random.seed(42)          # reproducible noise attacks

    # ══════════════════════════════════════════════════════════════════════════
    # 1.  Load images & embed DCT-SVD watermark
    # ══════════════════════════════════════════════════════════════════════════
    log("Loading images …")
    cover, watermark = load_images(COVER_PATH, WATERMARK_PATH, WM_SHAPE)
    log(f"  Cover: {cover.shape}   Watermark: {watermark.shape}")

    log("Embedding DCT-SVD watermark …")
    watermarked, key = embed_watermark(cover, watermark)

    psnr_val = psnr(cover, watermarked)
    ssim_val = ssim_metric(cover, watermarked)
    log(f"  PSNR = {psnr_val:.2f} dB  |  SSIM = {ssim_val:.4f}")

    save_image(watermarked, DIRS['watermarked'] / 'watermarked_lena.png')
    log("  Saved watermarked image.")

    # ── Task 1a : imperceptibility plot ───────────────────────────────────────
    save_cover_vs_watermarked(
        cover, watermarked, psnr_val, ssim_val,
        str(DIRS['watermarked'] / 'cover_vs_watermarked.png')
    )
    log("  Saved cover vs watermarked comparison plot.")

    # ── Task 1b : singular-value profiles ─────────────────────────────────────
    save_sv_profiles(
        svd_cover      = key['svd_cover'],
        original_svs   = key['original_svs'],
        modified_quads = key['modified_quads'],
        output_path    = str(DIRS['watermarked'] / 'singular_value_profiles.png')
    )
    log("  Saved singular-value profiles.")

    # ── PSNR/SSIM table ───────────────────────────────────────────────────────
    pd.DataFrame([{
        'Method': 'DCT-SVD',
        'PSNR (dB)': round(psnr_val, 4),
        'SSIM'     : round(ssim_val, 6),
    }]).to_csv(str(DIRS['tables'] / 'psnr_ssim.csv'), index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # 2.  Embed baseline watermark
    # ══════════════════════════════════════════════════════════════════════════
    log("Embedding pure-DCT baseline watermark …")
    watermarked_bl, key_bl = embed_baseline(cover, watermark)
    psnr_bl  = psnr(cover, watermarked_bl)
    ssim_bl  = ssim_metric(cover, watermarked_bl)
    log(f"  Baseline  PSNR = {psnr_bl:.2f} dB  |  SSIM = {ssim_bl:.4f}")
    save_image(watermarked_bl, DIRS['baseline'] / 'watermarked_lena_baseline.png')

    # Append to PSNR table
    psnr_rows = [
        {'Method': 'DCT-SVD',       'PSNR (dB)': round(psnr_val, 4), 'SSIM': round(ssim_val, 6)},
        {'Method': 'DCT Baseline',  'PSNR (dB)': round(psnr_bl,  4), 'SSIM': round(ssim_bl,  6)},
    ]
    pd.DataFrame(psnr_rows).to_csv(str(DIRS['tables'] / 'psnr_ssim.csv'), index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # 3.  No-attack extraction
    # ══════════════════════════════════════════════════════════════════════════
    log("Extracting watermarks — no attack …")
    ext_wms, sv_corrs, px_corrs, _ = extract_watermark(watermarked, key)
    out_dir = DIRS['extracted'] / 'no_attack'
    _save_quadrant_images(ext_wms, out_dir)
    # Note: Removed save_extracted_grid call to avoid individual files
    log(f"  SV correlations: {[f'{c:.4f}' for c in sv_corrs]}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4.  Apply each attack at default intensity; extract; collect results
    # ══════════════════════════════════════════════════════════════════════════
    all_attack_images = {}
    sv_corr_table     = {}     # { attack_key : [corr_B1..B4] }
    px_corr_table     = {}
    bl_px_corr_table  = {}

    # No-attack row
    sv_corr_table['no_attack'] = sv_corrs
    px_corr_table['no_attack'] = px_corrs
    bl_ext, bl_px, _           = extract_baseline(watermarked_bl, key_bl)
    bl_px_corr_table['no_attack'] = bl_px

    # Save no_attack baseline grid
    bl_out_no_attack = DIRS['baseline'] / 'extracted' / 'no_attack'
    bl_out_no_attack.mkdir(parents=True, exist_ok=True)
    _save_quadrant_images(bl_ext, bl_out_no_attack)
    save_extracted_grid(bl_ext, [0, 0, 0, 0], bl_px, 'No Attack',
                        str(bl_out_no_attack / 'grid.png'))

    log("\nRunning 12 attacks at default intensity …")
    for akey, default_param in ATTACK_DEFAULTS.items():
        label = ATTACK_DISPLAY[akey]
        log(f"  {label}  (param={default_param}) …", )

        # ── DCT-SVD ──────────────────────────────────────────────────────────
        attacked = apply_attack(akey, watermarked, default_param)
        all_attack_images[akey] = attacked
        save_image(attacked, DIRS['attacks'] / f'{akey}.png')

        ext_wms, sv_c, px_c, _ = extract_watermark(attacked, key)
        sv_corr_table[akey]     = sv_c
        px_corr_table[akey]     = px_c

        # Remove individual grid saving to save space
        # Only save quadrant images for potential individual analysis
        out_dir = DIRS['extracted'] / akey
        _save_quadrant_images(ext_wms, out_dir)
        # Note: Removed save_extracted_grid call to avoid individual files

        # ── Baseline ─────────────────────────────────────────────────────────
        attacked_bl       = apply_attack(akey, watermarked_bl, default_param)
        bl_ext, bl_px, _  = extract_baseline(attacked_bl, key_bl)
        bl_px_corr_table[akey] = bl_px
        bl_out = DIRS['baseline'] / 'extracted' / akey
        _save_quadrant_images(bl_ext, bl_out)
        # Save baseline grid for the merged image
        save_extracted_grid(bl_ext, [0, 0, 0, 0], bl_px, label,
                            str(bl_out / 'grid.png'))

    # ── Attack gallery ────────────────────────────────────────────────────────
    save_attack_gallery(all_attack_images,
                        str(DIRS['plots'] / 'attack_gallery.png'))

    # ── All attacks grid comparison ─────────────────────────────────────────────
    # Collect grid paths for all attacks (only attacks, excluding no_attack)
    all_grid_paths = {}
    bl_grid_paths = {}
    for akey in ATTACK_DEFAULTS.keys():
        label = ATTACK_DISPLAY[akey]
        # DCT-SVD grid paths
        grid_path = DIRS['extracted'] / akey / 'grid.png'
        all_grid_paths[akey] = str(grid_path)
        # Baseline grid paths
        bl_grid_path = DIRS['baseline'] / 'extracted' / akey / 'grid.png'
        bl_grid_paths[akey] = str(bl_grid_path)

    save_all_attacks_grid(
        all_grid_paths,
        str(DIRS['plots'] / 'all_attacks_grid.png')
    )

    # Save baseline all attacks grid comparison
    save_all_baseline_attacks_grid(
        bl_grid_paths,
        str(DIRS['plots'] / 'all_baseline_attacks_grid.png')
    )

    # ── Correlation heatmaps ──────────────────────────────────────────────────
    _save_heatmaps(sv_corr_table, px_corr_table, bl_px_corr_table)

    # ── Correlation CSV tables ────────────────────────────────────────────────
    _save_corr_tables(sv_corr_table, px_corr_table, bl_px_corr_table)

    # ── Baseline comparison plots ─────────────────────────────────────────────
    # Use attack keys only (not 'no_attack')
    attack_only_keys = list(ATTACK_DEFAULTS.keys())
    px_atk = {k: px_corr_table[k]    for k in attack_only_keys}
    bl_atk = {k: bl_px_corr_table[k] for k in attack_only_keys}

    save_baseline_comparison(
        px_atk, bl_atk,
        str(DIRS['plots'] / 'baseline_comparison.png')
    )
    save_baseline_comparison_all_quads(
        px_atk, bl_atk,
        str(DIRS['plots'] / 'baseline_comparison_all_quads.png')
    )
    log("  Saved baseline comparison plots.")

    # ══════════════════════════════════════════════════════════════════════════
    # 5.  Multi-intensity attack analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\nRunning multi-intensity attack analysis …")
    intensity_results_sv  = {}
    intensity_results_px  = {}
    intensity_rows        = []     # for CSV

    for akey, levels in ATTACK_LEVELS.items():
        label = ATTACK_DISPLAY[akey]
        log(f"  {label} …")
        sv_res, px_res = [], []

        for level in levels:
            attacked_i = apply_attack(akey, watermarked, level)
            _, sv_c, px_c, _ = extract_watermark(attacked_i, key)
            sv_res.append((level, sv_c))
            px_res.append((level, px_c))
            intensity_rows.append({
                'attack'  : akey,
                'level'   : level,
                **{f'sv_B{k+1}' : sv_c[k] for k in range(4)},
                **{f'px_B{k+1}' : px_c[k] for k in range(4)},
            })

        intensity_results_sv[akey] = sv_res
        intensity_results_px[akey] = px_res

        # Individual intensity curve (SV metric)
        save_intensity_curves_single(
            sv_res, akey,
            str(DIRS['plots'] / f'intensity_{akey}.png'),
            metric='sv'
        )

    # Combined 3×4 intensity figure
    save_intensity_curves_all(
        intensity_results_sv,
        str(DIRS['plots'] / 'intensity_curves_all.png'),
        metric='sv'
    )
    log("  Saved intensity curves.")

    # Intensity CSV
    pd.DataFrame(intensity_rows).to_csv(
        str(DIRS['tables'] / 'intensity_results.csv'), index=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    log("\nAll experiments complete.")
    log(f"Results saved to:  {OUTPUT_DIR.resolve()}/")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_quadrant_images(wm_list, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, wm in enumerate(wm_list):
        cv2.imwrite(str(out_dir / f'{QUAD_NAMES[k]}.png'), wm)


def _save_heatmaps(sv_table, px_table, bl_table) -> None:
    all_keys   = ['no_attack'] + list(ATTACK_DEFAULTS.keys())
    row_labels = ['No Attack'] + [ATTACK_DISPLAY[k] for k in ATTACK_DEFAULTS]

    sv_mat = np.array([sv_table[k]  for k in all_keys])
    px_mat = np.array([px_table[k]  for k in all_keys])
    bl_mat = np.array([bl_table[k]  for k in all_keys])

    save_correlation_heatmap(
        sv_mat, row_labels, QUAD_NAMES,
        title='DCT-SVD: SV Pearson Correlation (per attack × quadrant)',
        output_path=str(DIRS['plots'] / 'heatmap_sv_corr.png')
    )
    save_correlation_heatmap(
        px_mat, row_labels, QUAD_NAMES,
        title='DCT-SVD: Pixel Pearson Correlation (per attack × quadrant)',
        output_path=str(DIRS['plots'] / 'heatmap_pixel_corr.png')
    )
    save_correlation_heatmap(
        bl_mat, row_labels, QUAD_NAMES,
        title='Pure DCT Baseline: Pixel Pearson Correlation (per attack × quadrant)',
        output_path=str(DIRS['plots'] / 'heatmap_baseline_pixel_corr.png')
    )


def _save_corr_tables(sv_table, px_table, bl_table) -> None:
    def to_df(table, keys):
        rows = []
        for k in keys:
            rows.append({'Attack': 'No Attack' if k == 'no_attack'
                         else ATTACK_DISPLAY[k],
                         **{f'{QUAD_NAMES[i]}': round(table[k][i], 6)
                            for i in range(4)}})
        return pd.DataFrame(rows)

    all_keys = ['no_attack'] + list(ATTACK_DEFAULTS.keys())
    to_df(sv_table, all_keys).to_csv(
        str(DIRS['tables'] / 'sv_correlation_table.csv'), index=False)
    to_df(px_table, all_keys).to_csv(
        str(DIRS['tables'] / 'pixel_correlation_table.csv'), index=False)
    to_df(bl_table, all_keys).to_csv(
        str(DIRS['tables'] / 'baseline_pixel_corr.csv'), index=False)
    log("  Saved correlation tables.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
