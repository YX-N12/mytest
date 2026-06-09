"""
utils.py  –  2-D DCT / IDCT and vectorised zig-zag scanning.

All functions operate on 2-D numpy float64 arrays.
"""

import numpy as np
from scipy.fftpack import dct, idct


# ─────────────────────────────────────────────────────────────────────────────
# 2-D orthonormal DCT-II / IDCT-II
# ─────────────────────────────────────────────────────────────────────────────

def dct2d(img: np.ndarray) -> np.ndarray:
    """2-D orthonormal DCT-II (row-wise then column-wise)."""
    return dct(
        dct(img.astype(np.float64), norm='ortho', axis=0),
        norm='ortho', axis=1
    )


def idct2d(coeffs: np.ndarray) -> np.ndarray:
    """2-D orthonormal IDCT-II (column-wise then row-wise)."""
    return idct(
        idct(coeffs.astype(np.float64), norm='ortho', axis=1),
        norm='ortho', axis=0
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised zig-zag scanning
# ─────────────────────────────────────────────────────────────────────────────

def zigzag_scan_order(rows: int, cols: int):
    """
    Pre-compute the zig-zag scan index arrays for a (rows × cols) matrix.

    The scan starts at the DC component (0, 0) and ends at the highest
    frequency component (rows-1, cols-1), traversing anti-diagonals and
    reversing direction on every other diagonal – exactly the JPEG convention.

    Returns
    -------
    row_idx, col_idx : np.ndarray, shape (rows*cols,), dtype int32
        Integer index arrays so that  ``matrix[row_idx, col_idx]``  yields
        the zig-zag-ordered 1-D coefficient sequence.
    """
    r_flat = np.repeat(np.arange(rows, dtype=np.int32), cols)   # (rows*cols,)
    c_flat = np.tile  (np.arange(cols, dtype=np.int32), rows)   # (rows*cols,)
    diag   = r_flat + c_flat                                      # anti-diagonal index

    # Within each anti-diagonal:
    #   even diagonals → descending r  →  use (rows - 1 - r) as secondary key
    #   odd  diagonals → ascending  r  →  use r as secondary key
    secondary = np.where(diag % 2 == 0, rows - 1 - r_flat, r_flat)

    # Compound sort key: primary = diag (×rows to leave room), secondary = secondary
    sort_key = diag.astype(np.int64) * rows + secondary

    order   = np.argsort(sort_key, kind='stable')
    return r_flat[order], c_flat[order]


# ─────────────────────────────────────────────────────────────────────────────
# DCT coefficient ↔ quadrant mapping
# ─────────────────────────────────────────────────────────────────────────────

def dct_to_quadrants(
        dct_img  : np.ndarray,
        quad_shape: tuple,
        zz_order : tuple = None):
    """
    Scatter a (H × W) DCT matrix into 4 equal-size quadrant matrices by
    reading coefficients in zig-zag order (B1 = lowest freq, B4 = highest).

    Parameters
    ----------
    dct_img    : 2-D float array, shape (H, W)
    quad_shape : (qh, qw) – must satisfy 4 × qh × qw == H × W
    zz_order   : (row_idx, col_idx) from a previous call; computed if None

    Returns
    -------
    quadrants : list of 4 float arrays, each (qh, qw)
    zz_order  : (row_idx, col_idx) – cache and reuse across calls
    """
    rows, cols = dct_img.shape
    qh, qw     = quad_shape
    assert 4 * qh * qw == rows * cols, (
        f"Quadrant shape {quad_shape} is incompatible with image shape "
        f"{dct_img.shape}: 4×{qh}×{qw}={4*qh*qw} ≠ {rows}×{cols}={rows*cols}"
    )

    if zz_order is None:
        zz_order = zigzag_scan_order(rows, cols)

    ri, ci   = zz_order
    flat     = dct_img[ri, ci]                             # 1-D zig-zag sequence
    quarter  = qh * qw
    quadrants = [
        flat[k * quarter:(k + 1) * quarter].reshape(qh, qw)
        for k in range(4)
    ]
    return quadrants, zz_order


def quadrants_to_dct(
        quadrants      ,
        original_shape : tuple,
        zz_order       : tuple) -> np.ndarray:
    """
    Gather 4 quadrant matrices back into a (H × W) DCT coefficient matrix.

    Parameters
    ----------
    quadrants      : list of 4 float arrays, each (qh, qw)
    original_shape : (H, W) of the full DCT matrix
    zz_order       : (row_idx, col_idx) from ``dct_to_quadrants``

    Returns
    -------
    dct_img : 2-D float array, shape (H, W)
    """
    ri, ci  = zz_order
    flat    = np.concatenate([q.ravel() for q in quadrants])
    dct_img = np.zeros(original_shape, dtype=np.float64)
    dct_img[ri, ci] = flat
    return dct_img


# ─────────────────────────────────────────────────────────────────────────────
# Misc image helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_display(img: np.ndarray) -> np.ndarray:
    """Linearly stretch any float/int image to uint8 [0, 255]."""
    img = img.astype(np.float64)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-9:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Clip and cast to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)
