import os

# MUST be before importing numpy, onnxruntime, fastapi, etc.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_DYNAMIC"] = "FALSE"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from routes.upscale import router


app = FastAPI(
    title="Professional AI Upscaler",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_middleware(
    GZipMiddleware,
    minimum_size=1024
)


app.include_router(router)


@app.get("/")
def root():

    return {
        "status": "running",
        "engine": "ONNX Runtime",
        "model": "RealESRGAN General x4 v3"
    }