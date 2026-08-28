from __future__ import annotations

import os
from pathlib import Path


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "tmp"
PROGRESS_DIR = BASE_DIR / "progress"

for directory in (
    OUTPUT_DIR,
    TEMP_DIR,
    PROGRESS_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "Xhunta AI Image Upscaler"
APP_VERSION = "2.0.0"


# ============================================================
# CPU / RUNTIME
# ============================================================

CPU_THREADS = max(
    1,
    int(
        os.getenv(
            "CPU_THREADS",
            "1",
        )
    ),
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    str(CPU_THREADS),
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    str(CPU_THREADS),
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    str(CPU_THREADS),
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    str(CPU_THREADS),
)

os.environ.setdefault(
    "OMP_WAIT_POLICY",
    "PASSIVE",
)

os.environ.setdefault(
    "OMP_DYNAMIC",
    "FALSE",
)


# ============================================================
# MODEL PATHS
# ============================================================

# ------------------------------------------------------------
# FSRCNN x2
# ------------------------------------------------------------

FAST_X2_MODEL_PATH = str(
    MODELS_DIR / "FSRCNN_x2.pb"
)

FAST_X2_MODEL_NAME = "fsrcnn"
FAST_X2_SCALE = 2


# ------------------------------------------------------------
# Real-ESRGAN x4
# ------------------------------------------------------------

REAL_ESRGAN_MODEL_PATH = str(
    MODELS_DIR / "realesr-general-x4v3.onnx"
)

REAL_ESRGAN_SCALE = 4


# ============================================================
# MODEL COMPATIBILITY ALIASES
# ============================================================

# These aliases support older modules while the codebase is
# being consolidated around the explicit model names.

FAST_X2_MODEL_SCALE = FAST_X2_SCALE

REAL_ESRGAN_MODEL_SCALE = REAL_ESRGAN_SCALE


# ------------------------------------------------------------
# Legacy aliases
# ------------------------------------------------------------

# Existing inference/session modules may still import these.

MODEL_PATH = REAL_ESRGAN_MODEL_PATH
MODEL_SCALE = REAL_ESRGAN_SCALE


# ============================================================
# QUALITY TARGETS
# ============================================================

QUALITY_TARGETS = {
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
}

VALID_QUALITIES = tuple(
    QUALITY_TARGETS.keys()
)


# ============================================================
# QUALITY → MODEL STRATEGY
# ============================================================

QUALITY_MODEL = {
    "2k": {
        "model": "fast_x2",
        "scale": FAST_X2_SCALE,
        "passes": 1,
    },

    "4k": {
        "model": "best_x4",
        "scale": REAL_ESRGAN_SCALE,
        "passes": 1,
    },

    "8k": {
        "model": "best_x4",
        "scale": REAL_ESRGAN_SCALE,
        "passes": 1,
    },
}


# ============================================================
# OUTPUT LIMITS
# ============================================================

MAX_OUTPUT_DIMENSION = int(
    os.getenv(
        "MAX_OUTPUT_DIMENSION",
        "8192",
    )
)

MAX_OUTPUT_PIXELS = int(
    os.getenv(
        "MAX_OUTPUT_PIXELS",
        str(8192 * 8192),
    )
)


# ============================================================
# INPUT LIMITS
# ============================================================

MAX_IMAGE_PIXELS = int(
    os.getenv(
        "MAX_IMAGE_PIXELS",
        str(8192 * 8192),
    )
)

MAX_UPLOAD_SIZE = int(
    os.getenv(
        "MAX_UPLOAD_SIZE",
        str(25 * 1024 * 1024),
    )
)


# ============================================================
# TILE PROCESSING
# ============================================================

TILE_PAD = int(
    os.getenv(
        "TILE_PAD",
        "16",
    )
)


# ------------------------------------------------------------
# FSRCNN x2
# ------------------------------------------------------------

FAST_TILE_SMALL = int(
    os.getenv(
        "FAST_TILE_SMALL",
        "512",
    )
)

FAST_TILE_MEDIUM = int(
    os.getenv(
        "FAST_TILE_MEDIUM",
        "768",
    )
)

FAST_TILE_LARGE = int(
    os.getenv(
        "FAST_TILE_LARGE",
        "1024",
    )
)


# ------------------------------------------------------------
# Real-ESRGAN x4
# ------------------------------------------------------------

ESRGAN_TILE_SMALL = int(
    os.getenv(
        "ESRGAN_TILE_SMALL",
        "256",
    )
)

ESRGAN_TILE_MEDIUM = int(
    os.getenv(
        "ESRGAN_TILE_MEDIUM",
        "384",
    )
)

ESRGAN_TILE_LARGE = int(
    os.getenv(
        "ESRGAN_TILE_LARGE",
        "512",
    )
)


# ============================================================
# PROGRESS
# ============================================================

PROGRESS_TTL_SECONDS = int(
    os.getenv(
        "PROGRESS_TTL_SECONDS",
        "3600",
    )
)


# ============================================================
# JOB QUEUE
# ============================================================

MAX_WORKERS = max(
    1,
    int(
        os.getenv(
            "UPSCALE_MAX_WORKERS",
            "1",
        )
    ),
)

MAX_QUEUE = max(
    1,
    int(
        os.getenv(
            "UPSCALE_MAX_QUEUE",
            "20",
        )
    ),
)

# ============================================================
# OUTPUT FORMAT
# ============================================================

# PNG is deliberately preserved.
#
# This project must NOT automatically convert PNG input
# into JPEG because that destroys transparency.

OUTPUT_FORMAT = "PNG"

PNG_COMPRESS_LEVEL = int(
    os.getenv(
        "PNG_COMPRESS_LEVEL",
        "6",
    )
)

PNG_COMPRESS_LEVEL = max(
    0,
    min(
        9,
        PNG_COMPRESS_LEVEL,
    ),
)


# ============================================================
# JPEG
# ============================================================

JPEG_QUALITY = int(
    os.getenv(
        "JPEG_QUALITY",
        "95",
    )
)

JPEG_QUALITY = max(
    1,
    min(
        100,
        JPEG_QUALITY,
    ),
)


# ============================================================
# VALIDATION
# ============================================================

if FAST_X2_SCALE <= 0:
    raise RuntimeError(
        "FAST_X2_SCALE must be greater than zero."
    )

if REAL_ESRGAN_SCALE <= 0:
    raise RuntimeError(
        "REAL_ESRGAN_SCALE must be greater than zero."
    )

if MAX_OUTPUT_DIMENSION <= 0:
    raise RuntimeError(
        "MAX_OUTPUT_DIMENSION must be greater than zero."
    )

if MAX_OUTPUT_PIXELS <= 0:
    raise RuntimeError(
        "MAX_OUTPUT_PIXELS must be greater than zero."
    )

if MAX_IMAGE_PIXELS <= 0:
    raise RuntimeError(
        "MAX_IMAGE_PIXELS must be greater than zero."
    )

if MAX_UPLOAD_SIZE <= 0:
    raise RuntimeError(
        "MAX_UPLOAD_SIZE must be greater than zero."
    )


# ============================================================
# CONFIG SUMMARY
# ============================================================

def config_summary() -> dict:
    """
    Return a safe runtime configuration summary.
    """

    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,

        "cpu_threads": CPU_THREADS,

        "fast_x2_model": FAST_X2_MODEL_PATH,
        "fast_x2_model_name": FAST_X2_MODEL_NAME,
        "fast_x2_scale": FAST_X2_SCALE,

        "realesrgan_model": REAL_ESRGAN_MODEL_PATH,
        "realesrgan_scale": REAL_ESRGAN_SCALE,

        "quality_targets": QUALITY_TARGETS,
        "quality_model": QUALITY_MODEL,

        "max_image_pixels": MAX_IMAGE_PIXELS,
        "max_output_dimension": MAX_OUTPUT_DIMENSION,
        "max_output_pixels": MAX_OUTPUT_PIXELS,
        "max_upload_size": MAX_UPLOAD_SIZE,

        "output_dir": str(OUTPUT_DIR),
        "temp_dir": str(TEMP_DIR),
        "progress_dir": str(PROGRESS_DIR),

        "max_workers": MAX_WORKERS,
        "max_queue": MAX_QUEUE,

        "png_compress_level": PNG_COMPRESS_LEVEL,
        "jpeg_quality": JPEG_QUALITY,
    }