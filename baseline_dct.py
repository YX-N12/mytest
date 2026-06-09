"""
baseline_dct.py  –  Pure DCT watermarking baseline (no SVD step).

Embedding
---------
  1.  DCT of cover image.
  2.  Zig-zag split into 4 quadrants.
  3.  For each quadrant k:  Bk* = Bk + α_k × W_dct
      where W_dct is the DCT of the watermark (resized to quad_shape).
  4.  Reassemble + IDCT.

Extraction
----------
  1.  DCT of attacked image, zig-zag split.
  2.  For each quadrant k:  W_dct_k^hat = (Bk_att − Bk_orig) / α_k
  3.  IDCT → extracted watermark per quadrant.

This is structurally identical to the DCT-SVD scheme but replaces the
SVD step with direct coefficient addition, making it a fair ablation.
"""

import numpy as np
import cv2

from utils  import dct2d, idct2d, dct_to_quadrants, quadrants_to_dct, normalize_display
from config import ALPHA_BASELINE, QUAD_SHAPE, WM_SHAPE, QUAD_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_baseline(cover    : np.ndarray,
                   watermark: np.ndarray,
                   alpha    : dict = None,
                   quad_shape: tuple = QUAD_SHAPE):
    """
    Embed watermark into cover image using pure DCT (no SVD).

    Parameters
    ----------
    cover      : uint8 grayscale, shape (H, W)
    watermark  : uint8 grayscale, shape quad_shape  (already resized)
    alpha      : per-quadrant scaling dict; defaults to ALPHA_BASELINE
    quad_shape : (qh, qw)

    Returns
    -------
    watermarked : uint8 grayscale, same shape as cover
    key         : dict – extraction data
    """
    if alpha is None:
        alpha = ALPHA_BASELINE

    cover_f = cover.astype(np.float64)
    wm_f    = watermark.astype(np.float64)

    # ── DCT of cover + zigzag ─────────────────────────────────────────────────
    dct_cover       = dct2d(cover_f)
    quads, zz_order = dct_to_quadrants(dct_cover, quad_shape)

    # ── DCT of watermark ──────────────────────────────────────────────────────
    dct_wm = dct2d(wm_f)                # shape (qh, qw)

    # ── Embed: Bk* = Bk + α_k × W_dct ────────────────────────────────────────
    modified_quads = []
    for k, q in enumerate(quads):
        alpha_k = alpha[QUAD_NAMES[k]]
        modified_quads.append(q + alpha_k * dct_wm)

    # ── Reassemble + IDCT ─────────────────────────────────────────────────────
    dct_mod     = quadrants_to_dct(modified_quads, dct_cover.shape, zz_order)
    watermarked = np.clip(idct2d(dct_mod), 0, 255).astype(np.uint8)

    key = {
        'quads_original': quads,        # list[4] of original quadrant arrays
        'dct_wm'        : dct_wm,       # watermark DCT (qh × qw)
        'cover_shape'   : cover.shape,
        'quad_shape'    : quad_shape,
        'zz_order'      : zz_order,
        'alpha'         : alpha,
        'watermark'     : watermark,    # uint8 original watermark
    }
    return watermarked, key


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_baseline(attacked: np.ndarray, key: dict):
    """
    Extract watermarks from each quadrant of a (possibly attacked) baseline-
    watermarked image.

    Returns
    -------
    extracted_wms : list[4] of uint8 arrays (quad_shape), display-normalised
    pixel_corrs   : list[4] of float – Pearson r between extracted pixels
                    and original watermark pixels
    raw_extractions : list[4] of float64 arrays before normalisation
    """
    quads_orig = key['quads_original']
    dct_wm     = key['dct_wm']          # noqa – kept in key for reference
    quad_shape = key['quad_shape']
    zz_order   = key['zz_order']
    alpha      = key['alpha']
    watermark  = key['watermark']

    # ── DCT + zigzag of attacked image ────────────────────────────────────────
    dct_att      = dct2d(attacked.astype(np.float64))
    quads_att, _ = dct_to_quadrants(dct_att, quad_shape, zz_order)

    extracted_wms   = []
    pixel_corrs     = []
    raw_extractions = []

    wm_flat = watermark.flatten().astype(np.float64)

    for k, (q_att, q_orig) in enumerate(zip(quads_att, quads_orig)):
        alpha_k = alpha[QUAD_NAMES[k]]

        # Recover watermark DCT coefficients
        wm_dct_hat = (q_att - q_orig) / alpha_k

        # IDCT → spatial watermark
        wm_raw = idct2d(wm_dct_hat)
        raw_extractions.append(wm_raw)
        extracted_wms.append(normalize_display(wm_raw))

        # Pixel Pearson correlation
        wm_clipped = np.clip(wm_raw, 0, 255).flatten()
        corr = _pearson(wm_flat, wm_clipped)
        pixel_corrs.append(corr)

    return extracted_wms, pixel_corrs, raw_extractions


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if not np.isnan(r) else 0.0
