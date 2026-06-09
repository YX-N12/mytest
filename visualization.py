"""
visualization.py  –  All plotting and image-saving routines.

Design conventions
------------------
* White figure backgrounds; high-DPI PNG output (dpi=150).
* Quadrant colours  B1/B2/B3/B4  are drawn from config.QUAD_COLORS.
* All axis labels, titles, and annotations are in English.
* No plt.show() calls – every function saves to disk and closes the figure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
from pathlib import Path
from typing import List, Dict

from config import QUAD_COLORS, QUAD_NAMES, ATTACK_DISPLAY, ATTACK_PARAM_LABEL

# ── Shared style defaults ─────────────────────────────────────────────────────
DPI         = 150
STYLE_KWARGS = dict(facecolor='white')

plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'savefig.facecolor': 'white',
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'lines.linewidth'  : 2.0,
    'lines.markersize' : 6,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Cover vs watermarked comparison
# ─────────────────────────────────────────────────────────────────────────────

def save_cover_vs_watermarked(cover      : np.ndarray,
                               watermarked: np.ndarray,
                               psnr_val   : float,
                               ssim_val   : float,
                               output_path: str) -> None:
    """Side-by-side comparison of cover and watermarked images."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), **STYLE_KWARGS)

    axes[0].imshow(cover, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Cover (Lena)', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(watermarked, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(
        f'Watermarked Image\nPSNR = {psnr_val:.2f} dB  |  SSIM = {ssim_val:.4f}',
        fontweight='bold'
    )
    axes[1].axis('off')

    fig.suptitle('DCT-SVD Watermark Embedding — Imperceptibility Check',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Singular-value profiles (B1–B4)
# ─────────────────────────────────────────────────────────────────────────────

def save_sv_profiles(svd_cover  ,           # list[4] of (U, s, Vh)
                     original_svs,          # list[4] of 1-D arrays (original σ)
                     modified_quads,        # list[4] of modified quadrant arrays
                     output_path: str) -> None:
    """
    4-panel plot: singular-value magnitude profiles for B1–B4.
    Each panel overlays 'before embedding' (solid) and 'after embedding' (dashed).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), **STYLE_KWARGS)
    axes = axes.ravel()

    for k in range(4):
        ax        = axes[k]
        s_orig    = original_svs[k]

        # singular values of modified quadrant
        _, s_mod, _ = np.linalg.svd(modified_quads[k], full_matrices=False)

        x = np.arange(1, len(s_orig) + 1)
        ax.semilogy(x, s_orig, color=QUAD_COLORS[k],
                    linestyle='-',  label='Before embedding',  alpha=0.85)
        ax.semilogy(x, s_mod[:len(s_orig)], color=QUAD_COLORS[k],
                    linestyle='--', label='After embedding', alpha=0.75)

        ax.set_title(f'Quadrant {QUAD_NAMES[k]}  –  Singular Value Profile',
                     fontweight='bold')
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Magnitude (log scale)')
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle=':', alpha=0.4)
        ax.set_xlim(1, len(s_orig))

    fig.suptitle('Singular Value Profiles — DCT Quadrants B1 to B4',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Extracted-watermark grid  (2 × 2 panel for one attack condition)
# ─────────────────────────────────────────────────────────────────────────────

def save_extracted_grid(extracted_wms: List[np.ndarray],
                        sv_corrs      : List[float],
                        pixel_corrs   : List[float],
                        attack_label  : str,
                        output_path   : str) -> None:
    """2 × 2 grid of per-quadrant extracted watermarks with correlation values."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), **STYLE_KWARGS)
    axes = axes.ravel()

    for k in range(4):
        ax = axes[k]
        ax.imshow(extracted_wms[k], cmap='gray', vmin=0, vmax=255)
        ax.set_title(
            f'Quadrant {QUAD_NAMES[k]}\n'
            f'SV-Corr = {sv_corrs[k]:.4f}  |  Pixel-Corr = {pixel_corrs[k]:.4f}',
            fontsize=9, color=QUAD_COLORS[k], fontweight='bold'
        )
        ax.axis('off')

    fig.suptitle(f'Extracted Watermarks — {attack_label}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Attack-intensity robustness curves  (combined 3 × 4 figure)
# ─────────────────────────────────────────────────────────────────────────────

def save_intensity_curves_all(intensity_results: Dict,
                               output_path       : str,
                               metric            : str = 'sv') -> None:
    """
    3 × 4 subplot figure: for each of the 12 attacks, plot B1–B4 correlation
    vs attack intensity.

    Parameters
    ----------
    intensity_results : dict  { attack_key : [(param_val, [corr_B1..B4]), …] }
    metric            : 'sv' for SV Pearson, 'pixel' for pixel Pearson
    """
    attack_keys = list(intensity_results.keys())
    n_attacks   = len(attack_keys)
    ncols, nrows = 4, 3

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(20, 13),
                              **STYLE_KWARGS)
    axes = axes.ravel()

    for idx, akey in enumerate(attack_keys):
        ax      = axes[idx]
        results = intensity_results[akey]   # list of (param, [corrs])
        params  = [r[0] for r in results]
        corrs   = np.array([r[1] for r in results])  # shape (n_levels, 4)

        for k in range(4):
            ax.plot(params, corrs[:, k],
                    color=QUAD_COLORS[k],
                    marker='o',
                    label=QUAD_NAMES[k])

        ax.set_title(ATTACK_DISPLAY.get(akey, akey), fontweight='bold', fontsize=9)
        ax.set_xlabel(ATTACK_PARAM_LABEL.get(akey, 'Parameter'), fontsize=8)
        ax.set_ylabel('Pearson Correlation', fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, linestyle=':', alpha=0.4)

    # Hide any extra axes (if n_attacks < nrows*ncols)
    for idx in range(n_attacks, nrows * ncols):
        axes[idx].axis('off')

    metric_label = 'SV Pearson Correlation' if metric == 'sv' else 'Pixel Pearson Correlation'
    fig.suptitle(f'DCT-SVD Robustness vs Attack Intensity  ({metric_label})',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


def save_intensity_curves_single(intensity_results: list,
                                  attack_key        : str,
                                  output_path       : str,
                                  metric            : str = 'sv') -> None:
    """Single-attack intensity curve (one file per attack)."""
    params = [r[0] for r in intensity_results]
    corrs  = np.array([r[1] for r in intensity_results])   # (n_levels, 4)

    fig, ax = plt.subplots(figsize=(8, 5), **STYLE_KWARGS)

    for k in range(4):
        ax.plot(params, corrs[:, k],
                color=QUAD_COLORS[k],
                marker='o',
                label=QUAD_NAMES[k])

    ax.set_title(f'{ATTACK_DISPLAY.get(attack_key, attack_key)} — Robustness vs Intensity',
                 fontweight='bold')
    ax.set_xlabel(ATTACK_PARAM_LABEL.get(attack_key, 'Parameter'))
    ax.set_ylabel('Pearson Correlation')
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.legend(loc='lower left')
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DCT-SVD vs Baseline comparison
# ─────────────────────────────────────────────────────────────────────────────

def save_baseline_comparison(dctsv_pixel_corrs   : Dict,   # {attack: [B1..B4]}
                              baseline_pixel_corrs: Dict,   # {attack: [B1..B4]}
                              output_path         : str) -> None:
    """
    Side-by-side grouped bar chart comparing DCT-SVD and pure-DCT baseline
    maximum pixel correlation across attacks.
    """
    attack_keys  = list(dctsv_pixel_corrs.keys())
    n_attacks    = len(attack_keys)
    x            = np.arange(n_attacks)
    labels       = [ATTACK_DISPLAY.get(k, k) for k in attack_keys]

    # Use the maximum correlation across the 4 quadrants for each method
    dctsv_max    = [max(dctsv_pixel_corrs[k])    for k in attack_keys]
    baseline_max = [max(baseline_pixel_corrs[k]) for k in attack_keys]

    width = 0.35
    fig, ax = plt.subplots(figsize=(18, 6), **STYLE_KWARGS)

    bars1 = ax.bar(x - width/2, dctsv_max,    width,
                   label='DCT-SVD (best quadrant)',
                   color='#2980B9', alpha=0.88, edgecolor='white')
    bars2 = ax.bar(x + width/2, baseline_max, width,
                   label='Pure DCT Baseline (best quadrant)',
                   color='#E67E22', alpha=0.88, edgecolor='white')

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7.5)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Max Pixel Pearson Correlation')
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--', label='Threshold 0.5')
    ax.legend(loc='upper right')
    ax.set_title('Robustness Comparison: DCT-SVD vs Pure DCT Baseline\n'
                 '(Best quadrant pixel Pearson correlation, default attack intensity)',
                 fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    _save(fig, output_path)


def save_baseline_comparison_all_quads(dctsv_pixel_corrs   : Dict,
                                        baseline_pixel_corrs: Dict,
                                        output_path         : str) -> None:
    """
    Detailed 4-quadrant comparison: 2-row figure.
    Row 1: DCT-SVD B1–B4 pixel correlations per attack.
    Row 2: Baseline B1–B4 pixel correlations per attack.
    """
    attack_keys = list(dctsv_pixel_corrs.keys())
    n_attacks   = len(attack_keys)
    x           = np.arange(n_attacks)
    labels      = [ATTACK_DISPLAY.get(k, k) for k in attack_keys]
    width       = 0.2
    offsets     = [-1.5, -0.5, 0.5, 1.5]

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), **STYLE_KWARGS)

    for row_idx, (title, corr_dict) in enumerate([
        ('DCT-SVD Method', dctsv_pixel_corrs),
        ('Pure DCT Baseline', baseline_pixel_corrs),
    ]):
        ax = axes[row_idx]
        for k in range(4):
            vals = [corr_dict[akey][k] for akey in attack_keys]
            ax.bar(x + offsets[k] * width, vals, width,
                   color=QUAD_COLORS[k], label=QUAD_NAMES[k],
                   alpha=0.88, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Pixel Pearson Correlation')
        ax.set_ylim(-0.1, 1.15)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.legend(loc='upper right')
        ax.set_title(f'{title} — Per-Quadrant Pixel Correlation', fontweight='bold')
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)

    fig.suptitle('DCT-SVD vs Pure DCT Baseline — Per-Quadrant Robustness',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Correlation heatmap  (attacks × quadrants)
# ─────────────────────────────────────────────────────────────────────────────

def save_correlation_heatmap(corr_table : np.ndarray,
                              row_labels : List[str],
                              col_labels : List[str],
                              title      : str,
                              output_path: str) -> None:
    """
    Heatmap of a (n_attacks × n_quads) correlation matrix.

    corr_table : shape (n_attacks, n_quads)
    """
    fig, ax = plt.subplots(figsize=(8, max(5, len(row_labels) * 0.55)), **STYLE_KWARGS)

    im = ax.imshow(corr_table, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Pearson Correlation')

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontweight='bold')
    ax.set_yticklabels(row_labels)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val   = corr_table[i, j]
            color = 'black' if abs(val) < 0.7 else 'white'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=color, fontsize=8.5)

    ax.set_title(title, fontweight='bold', pad=10)
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Attack gallery  (all 12 attacked images on one canvas)
# ─────────────────────────────────────────────────────────────────────────────

def save_attack_gallery(attack_images: Dict[str, np.ndarray],
                         output_path  : str) -> None:
    """3 × 4 mosaic of all 12 attacked watermarked images."""
    attack_keys = list(attack_images.keys())
    n           = len(attack_keys)
    ncols, nrows = 4, (n + 3) // 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4), **STYLE_KWARGS)
    axes = np.array(axes).ravel()

    for idx, akey in enumerate(attack_keys):
        ax = axes[idx]
        ax.imshow(attack_images[akey], cmap='gray', vmin=0, vmax=255)
        ax.set_title(ATTACK_DISPLAY.get(akey, akey), fontsize=9, fontweight='bold')
        ax.axis('off')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Watermarked Image After Each Attack (Default Intensity)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, output_path)


def save_all_attacks_grid(attack_grid_paths: Dict[str, str],
                          output_path      : str) -> None:
    """
    3 × 4 grid showing extracted watermarks from all 12 attacks.
    Each cell contains the 2×2 grid of extracted watermarks for that attack.
    """
    attack_keys = list(attack_grid_paths.keys())
    n = len(attack_keys)
    ncols, nrows = 4, (n + 3) // 4

    # Calculate figure size: increased scale for better visibility
    fig_width = ncols * 4.5  # 4.5 inches per column (increased from 3.5)
    fig_height = nrows * 4.5  # 4.5 inches per row (increased from 3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), **STYLE_KWARGS)
    axes = np.array(axes).ravel()

    for idx, akey in enumerate(attack_keys):
        ax = axes[idx]

        # Load the grid image for this attack
        grid_path = Path(attack_grid_paths[akey])
        if grid_path.exists():
            grid_img = cv2.imread(str(grid_path))
            grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
            ax.imshow(grid_img)
            ax.set_title(ATTACK_DISPLAY.get(akey, akey), fontsize=10, fontweight='bold')
        else:
            # Placeholder if image doesn't exist
            ax.text(0.5, 0.5, f'No image\nfor\n{akey}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='red')
            ax.set_title(ATTACK_DISPLAY.get(akey, akey), fontsize=10, fontweight='bold')

        ax.axis('off')

    # Hide any extra axes (if n_attacks < nrows*ncols)
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('DCT-SVD Watermark Extraction — All Attacks Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout()
    _save(fig, output_path)


def save_all_baseline_attacks_grid(attack_grid_paths: Dict[str, str],
                                   output_path      : str) -> None:
    """
    3 × 4 grid showing baseline extracted watermarks from all 12 attacks.
    Each cell contains the 2×2 grid of extracted watermarks for that attack.
    """
    attack_keys = list(attack_grid_paths.keys())
    n = len(attack_keys)
    ncols, nrows = 4, (n + 3) // 4

    # Calculate figure size: same scale as DCT-SVD grid
    fig_width = ncols * 4.5
    fig_height = nrows * 4.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), **STYLE_KWARGS)
    axes = np.array(axes).ravel()

    for idx, akey in enumerate(attack_keys):
        ax = axes[idx]

        # Load the grid image for this attack
        grid_path = Path(attack_grid_paths[akey])
        if grid_path.exists():
            grid_img = cv2.imread(str(grid_path))
            grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
            ax.imshow(grid_img)
            ax.set_title(ATTACK_DISPLAY.get(akey, akey), fontsize=10, fontweight='bold')
        else:
            # Placeholder if image doesn't exist
            ax.text(0.5, 0.5, f'No image\nfor\n{akey}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='red')
            ax.set_title(ATTACK_DISPLAY.get(akey, akey), fontsize=10, fontweight='bold')

        ax.axis('off')

    # Hide any extra axes (if n_attacks < nrows*ncols)
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Pure DCT Baseline — Watermark Extraction — All Attacks Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Private helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    """Save figure as a high-resolution PNG with a white background and close it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
