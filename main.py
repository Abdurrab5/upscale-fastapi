import os


# ============================================================
# CPU ENVIRONMENT
# ============================================================

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    "1",
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
# FASTAPI
# ============================================================

from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.middleware.gzip import (
    GZipMiddleware,
)


# ============================================================
# APPLICATION CONFIG
# ============================================================

from config import (
    MAX_UPLOAD_SIZE,
    MAX_OUTPUT_DIMENSION,
    MAX_OUTPUT_PIXELS,
)


# ============================================================
# ROUTES
# ============================================================

from routes.upscale import router


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Xhunta AI Image Upscaler",
    version="2.2.0",
    description=(
        "Server-side fallback AI image upscaler "
        "for browser WebGPU processing failures."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=["*"],
)


# ============================================================
# GZIP
# ============================================================

app.add_middleware(
    GZipMiddleware,
    minimum_size=1024,
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(
    router
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",

        "service": (
            "Xhunta AI Image Upscaler"
        ),

        "version": "2.2.0",

        "engine": "ONNX Runtime",

        "model": (
            "RealESRGAN General x4 v3"
        ),

        "model_scale": 4,

        "processing": (
            "synchronous"
        ),

        "purpose": (
            "Server-side fallback when "
            "browser WebGPU processing fails."
        ),

        # ----------------------------------------------------
        # Public quality options
        # ----------------------------------------------------
        #
        # HD and 2K both resolve to a 2048px
        # final longest edge.
        #
        "quality_options": [
            "hd",
            "2k",
            "4k",
        ],

        "quality_targets": {
            "hd": 2048,
            "2k": 2048,
            "4k": 4096,
        },

        # ----------------------------------------------------
        # Supported output formats
        # ----------------------------------------------------

        "output_formats": [
            "png",
            "jpeg",
            "webp",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
def health():

    return {
        "status": "healthy",

        "service": (
            "Xhunta AI Image Upscaler"
        ),

        "engine": "ONNX Runtime",

        "model": (
            "RealESRGAN General x4 v3"
        ),
    }


# ============================================================
# CAPABILITIES
# ============================================================

@app.get(
    "/capabilities"
)
def capabilities():

    return {
        # ----------------------------------------------------
        # Quality options exposed to Laravel/frontend
        # ----------------------------------------------------

        "qualities": [
            "hd",
            "2k",
            "4k",
        ],

        # ----------------------------------------------------
        # Final longest-edge targets
        # ----------------------------------------------------

        "quality_targets": {
            "hd": 2048,
            "2k": 2048,
            "4k": 4096,
        },

        # ----------------------------------------------------
        # Upload limits
        # ----------------------------------------------------

        "max_upload_bytes": (
            MAX_UPLOAD_SIZE
        ),

        # ----------------------------------------------------
        # Output safety limits
        # ----------------------------------------------------

        "max_output_dimension": (
            MAX_OUTPUT_DIMENSION
        ),

        "max_output_pixels": (
            MAX_OUTPUT_PIXELS
        ),

        # ----------------------------------------------------
        # AI model
        # ----------------------------------------------------

        "model_scale": 4,

        "model": (
            "RealESRGAN General x4 v3"
        ),

        "engine": "ONNX Runtime",

        # ----------------------------------------------------
        # Input formats
        # ----------------------------------------------------

        "formats": [
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],

        # ----------------------------------------------------
        # Output formats
        # ----------------------------------------------------

        "output_formats": [
            "png",
            "jpeg",
            "webp",
        ],

        # ----------------------------------------------------
        # API behavior
        # ----------------------------------------------------

        "processing": (
            "synchronous"
        ),

        "endpoint": "/upscale",
    }