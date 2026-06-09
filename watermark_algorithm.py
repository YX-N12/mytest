"""
watermark_algorithm.py  –  DCT-SVD watermark embedding and extraction.

Algorithm (Sverdlov, Dexter & Eskicioglu):
  Embedding
    1.  2-D DCT of cover image.
    2.  Zig-zag scan  →  split coefficients into 4 quadrants B1..B4.
    3.  SVD of each quadrant:  Bk = Ua_k · Σa_k · Va_k^T
    4.  2-D DCT of (resized) watermark.
    5.  SVD of watermark DCT:  W_dct = Uw · Σw · Vw^T
    6.  Modify singular values:  σ*_i = σ_i + α_k · σw_i
    7.  Rebuild modified quadrants, reassemble, apply IDCT.

  Extraction
    1.  DCT of (possibly attacked) watermarked image.
    2.  Zig-zag  →  4 quadrants.
    3.  SVD of each attacked quadrant  →  σ*_i
    4.  Recover watermark SVs:  σw_i^hat = (σ*_i − σ_i) / α_k
    5.  Reconstruct watermark DCT:  W_dct^hat = Uw · diag(σw^hat) · Vw^T
    6.  IDCT  →  extracted watermark images.
"""

import numpy as np
import cv2

from utils  import dct2d, idct2d, dct_to_quadrants, quadrants_to_dct, normalize_display
from config import ALPHA, QUAD_SHAPE, WM_SHAPE, QUAD_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_images(cover_path: str, watermark_path: str, wm_shape=WM_SHAPE):
    """
    Load cover and watermark images as grayscale uint8 arrays.

    The watermark is resized to ``wm_shape`` (H, W) using area interpolation.

    Returns
    -------
    cover     : np.ndarray, uint8, shape COVER_SHAPE
    watermark : np.ndarray, uint8, shape wm_shape
    """
    cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    if cover is None:
        raise FileNotFoundError(f"Cannot load cover image: {cover_path}")

    wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Cannot load watermark: {watermark_path}")

    qh, qw = wm_shape
    wm_resized = cv2.resize(wm, (qw, qh), interpolation=cv2.INTER_AREA)
    return cover, wm_resized


# ─────────────────────────────────────────────────────────────────────────────
# Watermark embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_watermark(cover: np.ndarray,
                    watermark: np.ndarray,
                    alpha: dict = None,
                    quad_shape: tuple = QUAD_SHAPE):
    """
    Embed ``watermark`` into ``cover`` using the DCT-SVD scheme.

    Parameters
    ----------
    cover      : uint8 grayscale, shape (H, W)
    watermark  : uint8 grayscale, shape quad_shape
    alpha      : embedding-strength dict {'B1':…, 'B2':…, 'B3':…, 'B4':…}
    quad_shape : (qh, qw) – shape of each zig-zag quadrant

    Returns
    -------
    watermarked : uint8 grayscale, same shape as cover
    key         : dict – all data required for extraction
    """
    if alpha is None:
        alpha = ALPHA

    cover_f = cover.astype(np.float64)
    wm_f    = watermark.astype(np.float64)

    # ── Step 1-2 : DCT + zigzag split ────────────────────────────────────────
    dct_cover           = dct2d(cover_f)
    quads, zz_order     = dct_to_quadrants(dct_cover, quad_shape)

    # ── Step 3 : SVD of each quadrant ─────────────────────────────────────────
    svd_cover = []          # list of (U, s, Vh)
    for q in quads:
        U, s, Vh = np.linalg.svd(q, full_matrices=False)
        svd_cover.append((U, s, Vh))

    # ── Step 4-5 : DCT + SVD of watermark ────────────────────────────────────
    dct_wm      = dct2d(wm_f)
    Uw, sw, Vwh = np.linalg.svd(dct_wm, full_matrices=False)

    # ── Step 6-7 : modify singular values, rebuild quadrants ─────────────────
    modified_quads = []
    original_svs   = []          # σ_i for each quadrant  (stored in key)

    for k, (U, s, Vh) in enumerate(svd_cover):
        alpha_k = alpha[QUAD_NAMES[k]]
        n_sv    = min(len(s), len(sw))

        s_mod      = s.copy()
        s_mod[:n_sv] = s[:n_sv] + alpha_k * sw[:n_sv]

        modified_quads.append(U @ np.diag(s_mod) @ Vh)
        original_svs.append(s.copy())

    # ── Step 8-9 : reassemble + IDCT ─────────────────────────────────────────
    dct_wm_img  = quadrants_to_dct(modified_quads, dct_cover.shape, zz_order)
    wm_img_f    = idct2d(dct_wm_img)
    watermarked = np.clip(wm_img_f, 0, 255).astype(np.uint8)

    key = {
        'original_svs'  : original_svs,     # list[4] of 1-D arrays σ_i
        'Uw'            : Uw,               # left  singular vectors of W_dct
        'sw'            : sw,               # singular values of W_dct
        'Vwh'           : Vwh,              # right singular vectors of W_dct
        'cover_shape'   : cover.shape,
        'quad_shape'    : quad_shape,
        'zz_order'      : zz_order,
        'alpha'         : alpha,
        'watermark'     : watermark,        # original watermark (for pixel corr)
        'dct_wm'        : dct_wm,           # watermark DCT
        'quads_original': [q.copy() for q in quads],   # pre-embedding quadrants
        'modified_quads': modified_quads,   # post-embedding quadrants (for SV plot)
        'svd_cover'     : svd_cover,        # list[4] of (U, s, Vh) tuples
    }
    return watermarked, key


# ─────────────────────────────────────────────────────────────────────────────
# Watermark extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_watermark(attacked: np.ndarray, key: dict):
    """
    Extract watermarks from each of the 4 quadrants of a (possibly attacked)
    watermarked image.

    Parameters
    ----------
    attacked : uint8 grayscale image
    key      : dict returned by ``embed_watermark``

    Returns
    -------
    extracted_wms  : list[4] of uint8 arrays (quad_shape), display-normalised
    sv_corrs       : list[4] of float – Pearson corr between σw and σw^hat
    pixel_corrs    : list[4] of float – Pearson corr between watermark pixels
                     and extracted watermark pixels
    raw_extractions: list[4] of float64 arrays  (before clipping) for analysis
    """
    original_svs = key['original_svs']
    Uw           = key['Uw']
    sw           = key['sw']
    Vwh          = key['Vwh']
    quad_shape   = key['quad_shape']
    zz_order     = key['zz_order']
    alpha        = key['alpha']
    watermark    = key['watermark']

    # ── Steps 1-2 : DCT + zigzag of attacked image ───────────────────────────
    dct_att      = dct2d(attacked.astype(np.float64))
    quads_att, _ = dct_to_quadrants(dct_att, quad_shape, zz_order)

    extracted_wms   = []
    sv_corrs        = []
    pixel_corrs     = []
    raw_extractions = []

    wm_flat = watermark.flatten().astype(np.float64)

    for k, q_att in enumerate(quads_att):
        alpha_k  = alpha[QUAD_NAMES[k]]
        s_orig   = original_svs[k]

        # ── Step 3 : SVD of attacked quadrant ────────────────────────────────
        _, s_att, _ = np.linalg.svd(q_att, full_matrices=False)

        # ── Step 4 : recover watermark singular values ────────────────────────
        n_sv           = min(len(s_att), len(s_orig), len(sw))
        sw_hat         = np.zeros(len(sw))
        sw_hat[:n_sv]  = (s_att[:n_sv] - s_orig[:n_sv]) / alpha_k

        # ── Step 5-6 : reconstruct watermark ─────────────────────────────────
        wm_dct_hat = Uw @ np.diag(sw_hat) @ Vwh
        wm_raw     = idct2d(wm_dct_hat)          # float64
        raw_extractions.append(wm_raw)
        extracted_wms.append(normalize_display(wm_raw))

        # ── SV Pearson correlation ────────────────────────────────────────────
        n_corr  = min(len(sw), n_sv)
        sv_corr = _pearson(sw[:n_corr], sw_hat[:n_corr])
        sv_corrs.append(sv_corr)

        # ── Pixel Pearson correlation ─────────────────────────────────────────
        wm_norm  = np.clip(wm_raw, 0, 255)
        pix_corr = _pearson(wm_flat, wm_norm.flatten())
        pixel_corrs.append(pix_corr)

    return extracted_wms, sv_corrs, pixel_corrs, raw_extractions


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D arrays; returns 0.0 on degenerate input."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if not np.isnan(r) else 0.0
