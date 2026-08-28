 
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from config import (
    FAST_X2_MODEL_NAME,
    FAST_X2_MODEL_SCALE,
    FAST_X2_MODEL_PATH,
)


# ============================================================
# MODEL SINGLETON
# ============================================================

_model = None
_model_lock = threading.Lock()


# ============================================================
# MODEL
# ============================================================

def get_fast_model():
    """
    Lazily load FSRCNN x2 exactly once.

    Model:
        OpenCV DNN Super Resolution
        FSRCNN x2
    """

    global _model

    if _model is not None:
        return _model

    with _model_lock:

        if _model is not None:
            return _model

        if not os.path.isfile(
            FAST_X2_MODEL_PATH
        ):
            raise FileNotFoundError(
                "FSRCNN x2 model not found: "
                f"{FAST_X2_MODEL_PATH}"
            )

        if not hasattr(
            cv2,
            "dnn_superres",
        ):
            raise RuntimeError(
                "cv2.dnn_superres is unavailable. "
                "Install opencv-contrib-python."
            )

        try:

            model = (
                cv2.dnn_superres
                .DnnSuperResImpl_create()
            )

            model.readModel(
                FAST_X2_MODEL_PATH
            )

            model.setModel(
                FAST_X2_MODEL_NAME,
                FAST_X2_MODEL_SCALE,
            )

            _model = model

            print(
                "[FAST-SR] FSRCNN x2 loaded",
                flush=True,
            )

            return _model

        except Exception as exc:

            raise RuntimeError(
                "Failed to initialize FSRCNN x2."
            ) from exc


# ============================================================
# VALIDATION
# ============================================================

def _validate_rgb(
    image: np.ndarray,
) -> None:

    if image is None:
        raise ValueError(
            "FSRCNN input cannot be None."
        )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise TypeError(
            "FSRCNN input must be numpy.ndarray."
        )

    if image.ndim != 3:
        raise ValueError(
            "FSRCNN input must be HWC."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "FSRCNN requires RGB input."
        )


# ============================================================
# SINGLE FSRCNN PASS
# ============================================================

def run_fast_pass(
    image: np.ndarray,
) -> np.ndarray:

    _validate_rgb(
        image
    )

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255,
        ).astype(
            np.uint8,
        )

    image = np.ascontiguousarray(
        image
    )

    model = get_fast_model()

    # OpenCV DNN super-resolution expects BGR.
    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    try:

        output_bgr = model.upsample(
            bgr
        )

    except Exception as exc:

        raise RuntimeError(
            "FSRCNN x2 inference failed."
        ) from exc

    if output_bgr is None:
        raise RuntimeError(
            "FSRCNN returned no output."
        )

    if output_bgr.ndim != 3:
        raise RuntimeError(
            f"Invalid FSRCNN output: "
            f"{output_bgr.shape}"
        )

    output_rgb = cv2.cvtColor(
        output_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return np.ascontiguousarray(
        output_rgb,
        dtype=np.uint8,
    )


# ============================================================
# EXACT FINAL RESIZE
# ============================================================

def resize_fast_result(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError(
            "Target dimensions must be positive."
        )

    if (
        image.shape[1] == width
        and image.shape[0] == height
    ):
        return image

    result = cv2.resize(
        image,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    return np.ascontiguousarray(
        result,
        dtype=np.uint8,
    )
 
