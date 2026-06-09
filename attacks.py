"""
attacks.py  –  Twelve image-attack functions used to test watermark robustness.

Each function takes a uint8 grayscale numpy array and a parameter, and returns
a uint8 grayscale numpy array of the same spatial dimensions.

Attack registry ``ATTACKS`` maps attack keys to metadata dicts.
"""

import io
import numpy as np
import cv2
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Individual attack implementations
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Gaussian low-pass blur.  kernel_size must be an odd integer ≥ 1."""
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    k = max(k, 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)


def gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    """Additive zero-mean Gaussian noise with standard deviation ``std``."""
    noise = np.random.normal(0.0, float(std), img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def pixelation(img: np.ndarray, block_size: int) -> np.ndarray:
    """Pixelation: downsample by ``block_size`` then upsample (nearest)."""
    h, w  = img.shape
    bs    = max(int(block_size), 1)
    small = cv2.resize(img, (max(w // bs, 1), max(h // bs, 1)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    """JPEG lossy compression with quality factor 1–100."""
    quality = int(np.clip(quality, 1, 100))
    _, enc  = cv2.imencode('.jpg', img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def jpeg2000_compression(img: np.ndarray, compression_ratio: float) -> np.ndarray:
    """
    JPEG2000 lossy compression at the requested compression ratio (e.g. 10 → 10:1).
    Falls back to a heavy JPEG if the JPEG2000 codec is unavailable.
    """
    ratio = max(float(compression_ratio), 1.0)
    bpp   = 8.0 / ratio                          # target bits-per-pixel

    try:
        pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        buf     = io.BytesIO()
        pil_img.save(buf, format='JPEG2000',
                     quality_mode='rates', quality_layers=[bpp])
        buf.seek(0)
        result = np.array(Image.open(buf).convert('L'))
        return result.astype(np.uint8)
    except Exception:
        # Fallback: map ratio → JPEG quality (approximate equivalence)
        q = max(1, int(100 / ratio))
        return jpeg_compression(img, q)


def sharpening(img: np.ndarray, strength: float) -> np.ndarray:
    """Unsharp-mask sharpening.  strength ≥ 0 (0 = no change)."""
    s       = float(strength)
    blurred = cv2.GaussianBlur(img.astype(np.float64), (5, 5), sigmaX=0)
    sharp   = img.astype(np.float64) + s * (img.astype(np.float64) - blurred)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def rescaling(img: np.ndarray, scale: float) -> np.ndarray:
    """Downsample by ``scale`` then upsample back to original size (bicubic)."""
    h, w    = img.shape
    sc      = float(np.clip(scale, 0.01, 1.0))
    new_h   = max(int(h * sc), 2)
    new_w   = max(int(w * sc), 2)
    small   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def rotation(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by ``angle`` degrees; black-fill out-of-frame regions."""
    h, w  = img.shape
    cx, cy = w / 2, h / 2
    M     = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def symmetric_crop(img: np.ndarray, crop_percent: float) -> np.ndarray:
    """
    Crop ``crop_percent`` % from each side then zero-pad back to original size.
    Simulates loss of border content while preserving image dimensions.
    """
    h, w    = img.shape
    pct     = float(np.clip(crop_percent, 0, 49)) / 100.0
    dh, dw  = max(int(h * pct), 1), max(int(w * pct), 1)

    cropped = img[dh: h - dh, dw: w - dw]
    result  = np.zeros_like(img)
    result[dh: h - dh, dw: w - dw] = cropped
    return result


def contrast_adjustment(img: np.ndarray, adjustment: float) -> np.ndarray:
    """Add a constant brightness offset to every pixel."""
    return np.clip(img.astype(np.float64) + float(adjustment), 0, 255).astype(np.uint8)


def histogram_equalization(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE histogram equalization; ``clip_limit`` controls contrast limiting."""
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    return clahe.apply(img)


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction: out = (in / 255) ^ gamma × 255."""
    g     = max(float(gamma), 1e-4)
    table = (np.arange(256, dtype=np.float64) / 255.0) ** g * 255.0
    lut   = np.clip(table, 0, 255).astype(np.uint8)
    return lut[img]


# ─────────────────────────────────────────────────────────────────────────────
# Attack registry
# ─────────────────────────────────────────────────────────────────────────────

ATTACKS = {
    'gaussian_blur' : gaussian_blur,
    'gaussian_noise': gaussian_noise,
    'pixelation'    : pixelation,
    'jpeg'          : jpeg_compression,
    'jpeg2000'      : jpeg2000_compression,
    'sharpening'    : sharpening,
    'rescaling'     : rescaling,
    'rotation'      : rotation,
    'cropping'      : symmetric_crop,
    'contrast'      : contrast_adjustment,
    'histogram_eq'  : histogram_equalization,
    'gamma'         : gamma_correction,
}


def apply_attack(attack_key: str, img: np.ndarray, param) -> np.ndarray:
    """Convenience wrapper: apply an attack by its registry key."""
    if attack_key not in ATTACKS:
        raise KeyError(f"Unknown attack: '{attack_key}'. "
                       f"Valid keys: {list(ATTACKS.keys())}")
    return ATTACKS[attack_key](img, param)
