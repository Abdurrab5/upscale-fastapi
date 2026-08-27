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


from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.middleware.gzip import (
    GZipMiddleware,
)

from routes.upscale import router


app = FastAPI(
    title="Xhunta AI Image Upscaler",
    version="2.0.0",
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
        "engine": "ONNX Runtime",
        "model": "RealESRGAN General x4 v3",
        "model_scale": 4,
        "output_modes": [
            "auto",
            "hd",
            "2k",
            "4k",
            "8k",
        ],
    }