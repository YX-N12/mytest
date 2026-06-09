"""
metrics.py  –  Image-quality and watermark-robustness metrics.

Functions
---------
psnr(img1, img2)       – Peak Signal-to-Noise Ratio in dB
ssim_metric(img1, img2)– Structural Similarity Index Measure
pearson_corr(a, b)     – Pearson product-moment correlation
"""

import numpy as np
from skimage.metrics import structural_similarity as ski_ssim


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute PSNR between two uint8 grayscale images.

    Parameters
    ----------
    img1, img2 : np.ndarray  –  same shape, values in [0, 255]

    Returns
    -------
    float  –  PSNR in dB (inf if images are identical)
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_metric(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between two uint8 grayscale images.

    Returns
    -------
    float in [-1, 1]
    """
    val = ski_ssim(
        img1.astype(np.float64),
        img2.astype(np.float64),
        data_range=255.0
    )
    return float(val)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson product-moment correlation between two arrays (any shape).

    Returns 0.0 if either array is constant.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if not np.isnan(r) else 0.0
