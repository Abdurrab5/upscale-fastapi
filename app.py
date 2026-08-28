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

from config import (
    QUALITY_TARGETS,
)

from routes.upscale import router


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Xhunta AI Image Upscaler",
    version="3.0.0",
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
# HEALTH
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Xhunta AI Image Upscaler",
        "version": "3.0.0",
        "engine": "ONNX Runtime",
        "models": {
            "fast_x2": "Fast x2",
            "real_esrgan_x4": "Real-ESRGAN General x4 v3",
        },
        "output_modes": [
            "hd",
            "2k",
            "4k",
        ],
        "target_longest_edges": {
            key: value
            for key, value
            in QUALITY_TARGETS.items()
        },
    }