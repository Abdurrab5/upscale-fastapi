import os


def _int_setting(name, default, minimum, maximum):
    raw = os.getenv(name)
    value = int(raw) if raw is not None else int(default)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _bool_setting(name, default):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


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

TILE_SMALL = _int_setting("UPSCALE_TILE_SMALL", 128, 64, 512)

TILE_MEDIUM = _int_setting("UPSCALE_TILE_MEDIUM", 192, 64, 512)

TILE_LARGE = _int_setting("UPSCALE_TILE_LARGE", 256, 64, 512)

TILE_PAD = _int_setting("UPSCALE_TILE_PAD", 16, 1, 64)

if min(TILE_SMALL, TILE_MEDIUM, TILE_LARGE) <= TILE_PAD * 2:
    raise ValueError("Every configured tile size must exceed twice UPSCALE_TILE_PAD.")


# ============================================================
# CPU CONFIGURATION
# ============================================================

CPU_THREADS = min(
    _int_setting("UPSCALE_CPU_THREADS", 2, 1, 4),
    os.cpu_count() or 2,
)


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

OUTPUT_FORMAT = "PNG"

PNG_COMPRESS_LEVEL = _int_setting("UPSCALE_PNG_COMPRESS_LEVEL", 6, 0, 9)

ONNX_ENABLE_CPU_MEM_ARENA = _bool_setting("ONNX_ENABLE_CPU_MEM_ARENA", True)
ONNX_ENABLE_MEM_PATTERN = _bool_setting("ONNX_ENABLE_MEM_PATTERN", True)


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
