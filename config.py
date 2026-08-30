import os


# ============================================================
# APPLICATION PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# REAL-ESRGAN MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "realesr-general-x4v3.onnx",
)


# ============================================================
# MODEL SCALE
# ============================================================

MODEL_SCALE = 4


# ============================================================
# QUALITY TARGETS
# ============================================================
#
# These values represent the LONGEST EDGE of the
# FINAL OUTPUT IMAGE.
#
# HD  = 2048px
# 2K  = 2048px
# 4K  = 4096px
#
# HD is intentionally an alias of 2K.
#
# These values do NOT mean:
#
#     source × 2
#     source × 4
#
# Real-ESRGAN performs its internal x4 AI inference.
# The resulting image is then resized to the requested
# final target dimensions while preserving aspect ratio.
#

QUALITY_TARGETS = {
    "hd": 2048,
    "2k": 2048,
    "4k": 4096,
}


# ============================================================
# DEFAULT QUALITY
# ============================================================

DEFAULT_QUALITY = "4k"


# ============================================================
# ABSOLUTE OUTPUT SAFETY LIMITS
# ============================================================
#
# 4K is currently the maximum public output resolution.
#
# Maximum longest edge:
#
#     4096px
#
# Maximum output pixels:
#
#     4096 × 4096
#

MAX_OUTPUT_DIMENSION = 4096

MAX_OUTPUT_PIXELS = (
    4096 * 4096
)


# ============================================================
# TILE CONFIGURATION
# ============================================================
#
# Tile sizes used by the CPU Real-ESRGAN pipeline.
#
# The service can select the appropriate tile size based
# on source image resolution.
#

TILE_SMALL = 128

TILE_MEDIUM = 192

TILE_LARGE = 256

TILE_PAD = 16


# ============================================================
# CPU CONFIGURATION
# ============================================================

CPU_THREADS = min(
    os.cpu_count() or 2,
    2,
)


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

OUTPUT_FORMAT = "PNG"

PNG_COMPRESS_LEVEL = 6


# ============================================================
# SUPPORTED OUTPUT FORMATS
# ============================================================

SUPPORTED_OUTPUT_FORMATS = {
    "png",
    "jpeg",
    "webp",
}


# ============================================================
# UPLOAD LIMITS
# ============================================================

# Maximum uploaded file size:
# 10 MB

MAX_UPLOAD_SIZE = (
    10 * 1024 * 1024
)


# ============================================================
# IMAGE PIXEL LIMIT
# ============================================================
#
# Maximum source image resolution accepted by the
# server before AI processing.
#

MAX_IMAGE_PIXELS = 12_000_000