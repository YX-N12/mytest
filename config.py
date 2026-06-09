"""
config.py  –  Central configuration for DCT-SVD watermarking experiments.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
COVER_PATH     = './lena_640x480.png'
WATERMARK_PATH = './watermark.png'
OUTPUT_DIR     = Path('output')

DIRS = {
    'watermarked': OUTPUT_DIR / 'watermarked',
    'attacks'    : OUTPUT_DIR / 'attacks',
    'extracted'  : OUTPUT_DIR / 'extracted',
    'plots'      : OUTPUT_DIR / 'plots',
    'tables'     : OUTPUT_DIR / 'tables',
    'baseline'   : OUTPUT_DIR / 'baseline',
}

# ─── Image Dimensions ─────────────────────────────────────────────────────────
# Cover : 640 × 480  (W × H)  → numpy shape (480, 640)
# Each zigzag quadrant: 480/2 × 640/2 = 240 × 320  (H × W)
COVER_SHAPE = (480, 640)   # (H, W)
QUAD_SHAPE  = (240, 320)   # (H, W) — each of the 4 quadrants
WM_SHAPE    = (240, 320)   # watermark resized to this before embedding

# ─── DCT-SVD Embedding Strength ───────────────────────────────────────────────
ALPHA = {'B1': 0.25, 'B2': 0.01, 'B3': 0.01, 'B4': 0.01}

# ─── Baseline DCT: same α values, no SVD ──────────────────────────────────────
# Direct addition of scaled watermark DCT to each quadrant
ALPHA_BASELINE = {'B1': 0.25, 'B2': 0.01, 'B3': 0.01, 'B4': 0.01}

# ─── Quadrant Display Colours ─────────────────────────────────────────────────
QUAD_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
QUAD_NAMES  = ['B1', 'B2', 'B3', 'B4']

# ─── Attack Default Parameters (used in main results table) ───────────────────
ATTACK_DEFAULTS = {
    'gaussian_blur' : 5,
    'gaussian_noise': 20,
    'pixelation'    : 2,
    'jpeg'          : 30,
    'jpeg2000'      : 10,
    'sharpening'    : 3,
    'rescaling'     : 0.5,
    'rotation'      : 20,
    'cropping'      : 25,
    'contrast'      : -20,
    'histogram_eq'  : 2.0,
    'gamma'         : 0.6,
}

# ─── Attack Intensity Levels (for robustness-vs-strength curves) ──────────────
ATTACK_LEVELS = {
    'gaussian_blur' : [3, 5, 7, 9, 11],
    'gaussian_noise': [5, 10, 20, 30, 50],
    'pixelation'    : [2, 4, 8, 16, 32],
    'jpeg'          : [90, 70, 50, 30, 10],
    'jpeg2000'      : [2, 5, 10, 20, 50],
    'sharpening'    : [1, 2, 3, 5, 8],
    'rescaling'     : [0.9, 0.75, 0.5, 0.25, 0.1],
    'rotation'      : [5, 10, 20, 30, 45],
    'cropping'      : [5, 10, 15, 20, 25],
    'contrast'      : [-10, -20, -40, -60, -80],
    'histogram_eq'  : [1.0, 2.0, 4.0, 8.0, 16.0],
    'gamma'         : [0.3, 0.5, 0.8, 1.5, 2.0],
}

# ─── Human-readable labels ────────────────────────────────────────────────────
ATTACK_DISPLAY = {
    'gaussian_blur' : 'Gaussian Blur',
    'gaussian_noise': 'Gaussian Noise',
    'pixelation'    : 'Pixelation',
    'jpeg'          : 'JPEG Compression',
    'jpeg2000'      : 'JPEG2000',
    'sharpening'    : 'Sharpening',
    'rescaling'     : 'Rescaling',
    'rotation'      : 'Rotation',
    'cropping'      : 'Sym. Cropping',
    'contrast'      : 'Contrast Adj.',
    'histogram_eq'  : 'Histogram Eq.',
    'gamma'         : 'Gamma Correction',
}

ATTACK_PARAM_LABEL = {
    'gaussian_blur' : 'Kernel Size',
    'gaussian_noise': 'Std Dev (pixel)',
    'pixelation'    : 'Block Size',
    'jpeg'          : 'Quality Factor',
    'jpeg2000'      : 'Compression Ratio',
    'sharpening'    : 'Strength',
    'rescaling'     : 'Scale Factor',
    'rotation'      : 'Angle (°)',
    'cropping'      : 'Crop Percentage (%)',
    'contrast'      : 'Brightness Offset',
    'histogram_eq'  : 'CLAHE Clip Limit',
    'gamma'         : 'Gamma Value',
}
