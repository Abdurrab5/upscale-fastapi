from __future__ import annotations

import os
import threading
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort

from config import (
    CPU_THREADS,
    ESRGAN_TILE_LARGE,
    ESRGAN_TILE_MEDIUM,
    ESRGAN_TILE_SMALL,
    MAX_IMAGE_PIXELS,
    REAL_ESRGAN_MODEL_PATH,
    REAL_ESRGAN_MODEL_SCALE,
    TILE_PAD,
)

from services.fast_sr import (
    run_fast_pass,
    resize_fast_result,
)


# ============================================================
# TYPES
# ============================================================

ProgressCallback = Callable[[int], None]


# ============================================================
# REAL-ESRGAN SESSION
# ============================================================

_realesrgan_session = None
_realesrgan_lock = threading.Lock()


def get_realesrgan_session():
    global _realesrgan_session

    if _realesrgan_session is not None:
        return _realesrgan_session

    with _realesrgan_lock:

        if _realesrgan_session is not None:
            return _realesrgan_session

        if not os.path.isfile(
            REAL_ESRGAN_MODEL_PATH
        ):
            raise FileNotFoundError(
                "Real-ESRGAN model not found: "
                f"{REAL_ESRGAN_MODEL_PATH}"
            )

        options = ort.SessionOptions()

        options.intra_op_num_threads = max(
            1,
            int(CPU_THREADS),
        )

        options.inter_op_num_threads = 1

        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        options.log_severity_level = 3

        session = ort.InferenceSession(
            REAL_ESRGAN_MODEL_PATH,
            sess_options=options,
            providers=[
                "CPUExecutionProvider",
            ],
        )

        _realesrgan_session = session

        print(
            "[REAL-ESRGAN] ONNX session loaded",
            flush=True,
        )

        return session


# ============================================================
# TILE SIZE
# ============================================================

def _choose_tile_size(
    width: int,
    height: int,
) -> int:

    pixels = width * height

    if pixels <= 2_000_000:
        return ESRGAN_TILE_LARGE

    if pixels <= 8_000_000:
        return ESRGAN_TILE_MEDIUM

    return ESRGAN_TILE_SMALL


# ============================================================
# MODEL EXECUTION
# ============================================================

def _run_onnx(
    session,
    tensor: np.ndarray,
) -> np.ndarray:

    input_name = (
        session.get_inputs()[0].name
    )

    outputs = session.run(
        None,
        {
            input_name: tensor,
        },
    )

    if not outputs:
        raise RuntimeError(
            "Real-ESRGAN returned no output."
        )

    output = outputs[0]

    if output.ndim != 4:
        raise RuntimeError(
            "Unexpected Real-ESRGAN output shape: "
            f"{output.shape}"
        )

    return output


# ============================================================
# TENSOR → RGB
# ============================================================

def _tensor_to_rgb_uint8(
    tensor: np.ndarray,
) -> np.ndarray:
    """
    Convert NCHW float32 [0,1]
    to HWC uint8 RGB.
    """

    if tensor is None:
        raise ValueError(
            "Input tensor cannot be None."
        )

    if tensor.ndim != 4:
        raise ValueError(
            "Expected NCHW tensor, got "
            f"{tensor.shape}"
        )

    if tensor.shape[0] != 1:
        raise ValueError(
            "Only batch size 1 is supported."
        )

    array = tensor[0]

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    array = (
        array * 255.0
    ).round().astype(
        np.uint8
    )

    array = array.transpose(
        1,
        2,
        0,
    )

    return np.ascontiguousarray(
        array,
        dtype=np.uint8,
    )


# ============================================================
# RGB → MODEL TENSOR
# ============================================================

def _rgb_to_tensor(
    image: np.ndarray,
) -> np.ndarray:

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if image.ndim != 3:
        raise ValueError(
            "Expected HWC image, got "
            f"{image.shape}"
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected RGB image with 3 channels."
        )

    if image.dtype != np.uint8:

        image = np.clip(
            image,
            0,
            255,
        ).astype(
            np.uint8,
            copy=False,
        )

    image = np.ascontiguousarray(
        image
    )

    tensor = (
        image.astype(
            np.float32
        ) / 255.0
    )

    tensor = tensor.transpose(
        2,
        0,
        1,
    )

    tensor = np.expand_dims(
        tensor,
        axis=0,
    )

    return np.ascontiguousarray(
        tensor,
        dtype=np.float32,
    )


# ============================================================
# MODEL OUTPUT → RGB
# ============================================================

def _model_output_to_rgb(
    output: np.ndarray,
) -> np.ndarray:

    if output.ndim != 4:
        raise ValueError(
            "Expected NCHW model output."
        )

    if output.shape[0] != 1:
        raise ValueError(
            "Only batch size 1 is supported."
        )

    output = output[0]

    output = np.clip(
        output,
        0.0,
        1.0,
    )

    output = (
        output * 255.0
    ).round().astype(
        np.uint8
    )

    output = output.transpose(
        1,
        2,
        0,
    )

    return np.ascontiguousarray(
        output,
        dtype=np.uint8,
    )


# ============================================================
# PAD TILE
# ============================================================

def _pad_tile(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    pad: int,
):

    height, width = image.shape[:2]

    x0 = max(
        0,
        left - pad,
    )

    y0 = max(
        0,
        top - pad,
    )

    x1 = min(
        width,
        right + pad,
    )

    y1 = min(
        height,
        bottom + pad,
    )

    tile = image[
        y0:y1,
        x0:x1,
    ]

    pad_left = max(
        0,
        pad - left,
    )

    pad_top = max(
        0,
        pad - top,
    )

    pad_right = max(
        0,
        (right + pad) - width,
    )

    pad_bottom = max(
        0,
        (bottom + pad) - height,
    )

    if (
        pad_left
        or pad_top
        or pad_right
        or pad_bottom
    ):

        tile = np.pad(
            tile,
            (
                (
                    pad_top,
                    pad_bottom,
                ),
                (
                    pad_left,
                    pad_right,
                ),
                (
                    0,
                    0,
                ),
            ),
            mode="reflect",
        )

    crop_left = (
        left - x0 + pad_left
    )

    crop_top = (
        top - y0 + pad_top
    )

    crop_right = (
        crop_left
        + (right - left)
    )

    crop_bottom = (
        crop_top
        + (bottom - top)
    )

    return (
        tile,
        crop_left,
        crop_top,
        crop_right,
        crop_bottom,
    )


# ============================================================
# REAL-ESRGAN TILE PIPELINE
# ============================================================

def run_realesrgan(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:

    if image is None:
        raise ValueError(
            "Real-ESRGAN input cannot be None."
        )

    if image.ndim != 3:
        raise ValueError(
            "Real-ESRGAN input must be HWC."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Real-ESRGAN requires RGB input."
        )

    source_height = image.shape[0]
    source_width = image.shape[1]

    source_pixels = (
        source_width * source_height
    )

    if source_pixels > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Input image exceeds the maximum "
            f"allowed size of {MAX_IMAGE_PIXELS:,} pixels."
        )

    session = get_realesrgan_session()

    tile_size = _choose_tile_size(
        source_width,
        source_height,
    )

    scale = int(
        REAL_ESRGAN_MODEL_SCALE
    )

    native_width = (
        source_width * scale
    )

    native_height = (
        source_height * scale
    )

    output = np.empty(
        (
            native_height,
            native_width,
            3,
        ),
        dtype=np.uint8,
    )

    tiles_x = (
        source_width
        + tile_size
        - 1
    ) // tile_size

    tiles_y = (
        source_height
        + tile_size
        - 1
    ) // tile_size

    total_tiles = (
        tiles_x * tiles_y
    )

    completed_tiles = 0

    print(
        "[REAL-ESRGAN] "
        f"Input: {source_width}x{source_height}",
        flush=True,
    )

    print(
        "[REAL-ESRGAN] "
        f"Native output: "
        f"{native_width}x{native_height}",
        flush=True,
    )

    print(
        "[REAL-ESRGAN] "
        f"Tile size: {tile_size}",
        flush=True,
    )

    print(
        "[REAL-ESRGAN] "
        f"Tiles: {total_tiles}",
        flush=True,
    )

    if progress_callback:
        progress_callback(1)

    # --------------------------------------------------------
    # PROCESS TILES
    # --------------------------------------------------------

    for top in range(
        0,
        source_height,
        tile_size,
    ):

        for left in range(
            0,
            source_width,
            tile_size,
        ):

            right = min(
                source_width,
                left + tile_size,
            )

            bottom = min(
                source_height,
                top + tile_size,
            )

            (
                tile,
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
            ) = _pad_tile(
                image,
                left,
                top,
                right,
                bottom,
                TILE_PAD,
            )

            tensor = _rgb_to_tensor(
                tile
            )

            model_output = _run_onnx(
                session,
                tensor,
            )

            result = _model_output_to_rgb(
                model_output
            )

            del model_output
            del tensor
            del tile

            # ------------------------------------------------
            # Remove model padding.
            # ------------------------------------------------

            crop_left = int(
                crop_left * scale
            )

            crop_top = int(
                crop_top * scale
            )

            crop_right = int(
                crop_right * scale
            )

            crop_bottom = int(
                crop_bottom * scale
            )

            result = result[
                crop_top:crop_bottom,
                crop_left:crop_right,
            ]

            output_left = (
                left * scale
            )

            output_top = (
                top * scale
            )

            output_right = (
                right * scale
            )

            output_bottom = (
                bottom * scale
            )

            expected_width = (
                output_right
                - output_left
            )

            expected_height = (
                output_bottom
                - output_top
            )

            if (
                result.shape[1]
                != expected_width
                or result.shape[0]
                != expected_height
            ):

                result = cv2.resize(
                    result,
                    (
                        expected_width,
                        expected_height,
                    ),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            output[
                output_top:output_bottom,
                output_left:output_right,
            ] = result

            del result

            completed_tiles += 1

            percent = int(
                (
                    completed_tiles
                    / total_tiles
                ) * 100
            )

            if progress_callback:
                progress_callback(
                    percent
                )

    # --------------------------------------------------------
    # EXACT TARGET SIZE
    # --------------------------------------------------------

    if (
        native_width == int(target_width)
        and native_height == int(target_height)
    ):
        return output

    final = cv2.resize(
        output,
        (
            int(target_width),
            int(target_height),
        ),
        interpolation=cv2.INTER_LANCZOS4,
    )

    del output

    return np.ascontiguousarray(
        final,
        dtype=np.uint8,
    )


# ============================================================
# ENGINE
# ============================================================

class SuperResolutionEngine:
    """
    Unified super-resolution engine.

    fast_x2:
        FSRCNN x2.

    best_x4:
        Real-ESRGAN x4.

    multi_stage_ai:
        Real-ESRGAN x4 followed by exact final resize.
    """

    def fast_x2(
        self,
        image: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> np.ndarray:

        # image is HWC RGB uint8 here.
        result = run_fast_pass(
            image
        )

        return resize_fast_result(
            result,
            target_width,
            target_height,
        )

    def realesrgan_x4(
        self,
        image,
        target_width: int,
        target_height: int,
        progress_callback=None,
    ) -> np.ndarray:

        # ----------------------------------------------------
        # load_image() supplies NCHW float32 [0,1].
        # Real-ESRGAN pipeline expects HWC uint8 RGB.
        # ----------------------------------------------------

        rgb = _tensor_to_rgb_uint8(
            image
        )

        return run_realesrgan(
            rgb,
            target_width,
            target_height,
            progress_callback=progress_callback,
        )

    def upscale(
        self,
        image,
        target_width: int,
        target_height: int,
        strategy: str,
        ai_passes: int = 1,
        progress_callback=None,
    ):

        strategy = (
            str(strategy or "")
            .strip()
            .lower()
        )

        if strategy == "fast_x2":

            rgb = _tensor_to_rgb_uint8(
                image
            )

            return self.fast_x2(
                rgb,
                target_width,
                target_height,
            )

        if strategy == "best_x4":

            return self.realesrgan_x4(
                image,
                target_width,
                target_height,
                progress_callback=progress_callback,
            )

        if strategy == "multi_stage_ai":

            return self.realesrgan_x4(
                image,
                target_width,
                target_height,
                progress_callback=progress_callback,
            )

        raise ValueError(
            "Unsupported upscale strategy: "
            f"{strategy}"
        )


# ============================================================
# GLOBAL ENGINE
# ============================================================

_engine = None
_engine_lock = threading.Lock()


def get_engine():

    global _engine

    if _engine is not None:
        return _engine

    with _engine_lock:

        if _engine is not None:
            return _engine

        _engine = (
            SuperResolutionEngine()
        )

        return _engine