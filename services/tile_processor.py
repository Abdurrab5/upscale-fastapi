 
import numpy as np
import time

from PIL import Image

from config import (
    MODEL_SCALE,
    TILE_LARGE,
    TILE_MEDIUM,
    TILE_PAD,
    TILE_SMALL,
)


# ============================================================
# TILE SIZE
# ============================================================

def choose_tile_size(
    width: int,
    height: int,
) -> int:
    """
    Select a CPU-friendly tile size based on source image area.

    Smaller images:
        smaller tiles → lower peak memory

    Larger images:
        larger tiles → fewer ONNX inference calls
    """

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError(
            "Invalid image dimensions."
        )

    pixels = width * height

    if pixels <= 2_000_000:
        return TILE_SMALL

    if pixels <= 6_000_000:
        return TILE_MEDIUM

    return TILE_LARGE


def _tile_axis_coordinates(length: int, tile_size: int, stride: int):
    """Yield the exact start/end coordinates used along one image axis."""

    for start in range(0, length, stride):
        end = min(start + tile_size, length)
        yield start, end
        if end >= length:
            break


def tile_count(width: int, height: int) -> int:
    tile_size = choose_tile_size(width, height)
    stride = tile_size - TILE_PAD * 2
    columns = sum(1 for _ in _tile_axis_coordinates(int(width), tile_size, stride))
    rows = sum(1 for _ in _tile_axis_coordinates(int(height), tile_size, stride))
    return columns * rows


# ============================================================
# GENERATE TILES
# ============================================================

def generate_tiles(
    image,
):
    """
    Generate overlapping HWC uint8 or NCHW tensor tiles lazily.

    Yields:

        tile,
        left,
        top,
        right,
        bottom

    Coordinates are relative to the original input image.

    The overlap is intentionally larger than zero so that
    neighboring Real-ESRGAN tiles can be cropped before being
    placed into the final canvas.
    """

    if image is None:
        raise ValueError(
            "Image tensor cannot be None."
        )

    if getattr(image, "ndim", 0) not in (3, 4):
        raise ValueError(
            "Image must have HWC uint8 or NCHW tensor shape."
        )

    if image.ndim == 3:
        height, width, channels = image.shape
    else:
        _, channels, height, width = image.shape

    if channels != 3:
        raise ValueError(
            "Real-ESRGAN expects an RGB tensor with 3 channels."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Invalid image dimensions."
        )

    tile_size = choose_tile_size(
        width,
        height,
    )

    # --------------------------------------------------------
    # TILE OVERLAP
    # --------------------------------------------------------

    overlap = TILE_PAD * 2

    if tile_size <= overlap:
        raise ValueError(
            "Tile size must be larger than the configured overlap."
        )

    stride = tile_size - overlap

    # --------------------------------------------------------
    # GENERATE TILE POSITIONS
    # --------------------------------------------------------

    for top, bottom in _tile_axis_coordinates(height, tile_size, stride):
        for left, right in _tile_axis_coordinates(width, tile_size, stride):

            if image.ndim == 3:
                tile = image[top:bottom, left:right, :]
            else:
                tile = image[:, :, top:bottom, left:right]

            yield (
                tile,
                left,
                top,
                right,
                bottom,
            )



# ============================================================
# MODEL OUTPUT → RGB ARRAY
# ============================================================

def _model_to_image(
    prediction,
):
    """
    Convert Real-ESRGAN ONNX output:

        NCHW float32 [0, 1]

    into:

        HWC uint8 [0, 255]
    """

    if prediction is None:
        raise RuntimeError(
            "Real-ESRGAN returned no output."
        )

    if prediction.ndim != 4:
        raise RuntimeError(
            "Unexpected ONNX model output shape: "
            f"{prediction.shape}"
        )

    if prediction.shape[0] != 1:
        raise RuntimeError(
            "Real-ESRGAN output batch size must be 1."
        )

    if prediction.shape[1] != 3:
        raise RuntimeError(
            "Real-ESRGAN output must contain 3 RGB channels."
        )

    # --------------------------------------------------------
    # NCHW → HWC
    # --------------------------------------------------------

    output = prediction[0]

    output = output.transpose(
        1,
        2,
        0,
    )

    # --------------------------------------------------------
    # Clamp model output
    # --------------------------------------------------------

    np.clip(
        output,
        0.0,
        1.0,
        out=output,
    )

    # --------------------------------------------------------
    # Float → uint8
    # --------------------------------------------------------

    output = (
        output * 255.0
    ).astype(
        np.uint8
    )

    return output


# ============================================================
# RESIZE RGB ARRAY
# ============================================================

def resize_image_array(
    array,
    width: int,
    height: int,
):
    """
    Resize an HWC uint8 RGB array using high-quality Lanczos
    resampling.

    If the requested dimensions already match, the original
    array is returned unchanged.
    """

    if array is None:
        raise ValueError(
            "Image array cannot be None."
        )

    if getattr(array, "ndim", 0) != 3:
        raise ValueError(
            "Image array must have HWC shape."
        )

    if array.shape[2] != 3:
        raise ValueError(
            "Image array must contain 3 RGB channels."
        )

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError(
            "Invalid resize dimensions."
        )

    if (
        array.shape[1] == width
        and array.shape[0] == height
    ):
        return array

    image = Image.fromarray(
        array,
        mode="RGB",
    )

    try:

        resized = image.resize(
            (
                width,
                height,
            ),
            Image.Resampling.LANCZOS,
        )

        return np.asarray(
            resized,
            dtype=np.uint8,
        ).copy()

    finally:

        image.close()


# ============================================================
# TILE CROP
# ============================================================

def _crop_ai_tile(
    prediction,
    left: int,
    top: int,
    right: int,
    bottom: int,
    source_width: int,
    source_height: int,
):
    """
    Remove the overlap area from a Real-ESRGAN x4 tile.

    TILE_PAD is expressed in source-image pixels, therefore
    the crop must be multiplied by MODEL_SCALE in the AI
    output.
    """

    crop_left = (
        TILE_PAD
        if left > 0
        else 0
    )

    crop_top = (
        TILE_PAD
        if top > 0
        else 0
    )

    crop_right = (
        TILE_PAD
        if right < source_width
        else 0
    )

    crop_bottom = (
        TILE_PAD
        if bottom < source_height
        else 0
    )

    x1 = crop_left * MODEL_SCALE
    y1 = crop_top * MODEL_SCALE

    x2 = (
        prediction.shape[1]
        - crop_right * MODEL_SCALE
    )

    y2 = (
        prediction.shape[0]
        - crop_bottom * MODEL_SCALE
    )

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            "Invalid Real-ESRGAN tile crop."
        )

    cropped = prediction[
        y1:y2,
        x1:x2,
    ]

    return (
        cropped,
        left + crop_left,
        top + crop_top,
        right - crop_right,
        bottom - crop_bottom,
    )


# ============================================================
# MAP SOURCE REGION → FINAL OUTPUT
# ============================================================

def _map_source_to_output(
    source_left: int,
    source_top: int,
    source_right: int,
    source_bottom: int,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
):
    """
    Map source-image coordinates to final output coordinates.

    Aspect ratio is preserved by the target resolver before this
    function is called.
    """

    target_left = int(
        round(
            source_left
            * output_width
            / source_width
        )
    )

    target_top = int(
        round(
            source_top
            * output_height
            / source_height
        )
    )

    target_right = int(
        round(
            source_right
            * output_width
            / source_width
        )
    )

    target_bottom = int(
        round(
            source_bottom
            * output_height
            / source_height
        )
    )

    target_left = max(
        0,
        min(
            output_width,
            target_left,
        ),
    )

    target_top = max(
        0,
        min(
            output_height,
            target_top,
        ),
    )

    target_right = max(
        target_left,
        min(
            output_width,
            target_right,
        ),
    )

    target_bottom = max(
        target_top,
        min(
            output_height,
            target_bottom,
        ),
    )

    return (
        target_left,
        target_top,
        target_right,
        target_bottom,
    )


# ============================================================
# SINGLE REAL-ESRGAN AI PASS
# ============================================================

def run_ai_pass(
    session,
    image,
    output_width: int,
    output_height: int,
    progress_callback=None,
    output_writer=None,
    stage_metrics=None,
):
    """
    Run one native Real-ESRGAN x4 pass.

    IMPORTANT ARCHITECTURE:

        Source image
             ↓
        overlapping tiles
             ↓
        Real-ESRGAN x4
             ↓
        crop tile overlap
             ↓
        resize/map into FINAL target
             ↓
        final output canvas

    The function NEVER creates a full source × 4 intermediate
    image.

    Example:

        Source:
            1000 × 750

        Requested:
            4096 longest edge

        Final:
            4096 × 3072

    Real-ESRGAN still internally produces x4 tile outputs, but
    those outputs are immediately reduced/mapped into the final
    requested resolution.
    """

    if session is None:
        raise ValueError(
            "ONNX session is required."
        )

    if image is None:
        raise ValueError(
            "Image tensor is required."
        )

    if getattr(image, "ndim", 0) == 3:
        source_height, source_width, channels = image.shape
    elif getattr(image, "ndim", 0) == 4:
        _, channels, source_height, source_width = image.shape
    else:
        raise ValueError("AI input must be an HWC image or NCHW tensor.")

    if channels != 3:
        raise ValueError(
            "AI input must contain 3 RGB channels."
        )

    source_width = int(source_width)
    source_height = int(source_height)

    output_width = int(
        output_width
    )

    output_height = int(
        output_height
    )

    if source_width <= 0 or source_height <= 0:
        raise ValueError(
            "Invalid AI input dimensions."
        )

    if output_width <= 0 or output_height <= 0:
        raise ValueError(
            "Invalid AI output dimensions."
        )

    # ========================================================
    # ONNX INPUT
    # ========================================================

    inputs = session.get_inputs()

    if not inputs:
        raise RuntimeError(
            "ONNX model has no input."
        )

    input_name = inputs[0].name

    # ========================================================
    # FINAL OUTPUT CANVAS
    # ========================================================
    #
    # This is ALWAYS the requested final resolution.
    #
    # Never source × 4.
    #

    canvas = None if output_writer is not None else np.zeros(
        (output_height, output_width, 3), dtype=np.uint8
    )

    # ========================================================
    # TILE COUNT
    # ========================================================

    total_tiles = tile_count(source_width, source_height)

    processed = 0

    # ========================================================
    # PROCESS TILES
    # ========================================================

    for (
        tile,
        left,
        top,
        right,
        bottom,
    ) in generate_tiles(image):

        prediction = None
        composition_started = None

        try:

            # =================================================
            # REAL-ESRGAN INFERENCE
            # =================================================

            if tile.ndim == 3:
                model_tile = np.ascontiguousarray(
                    tile.transpose(2, 0, 1)[None, ...], dtype=np.float32
                )
                model_tile *= (1.0 / 255.0)
            else:
                model_tile = tile

            inference_started = time.perf_counter()
            prediction = session.run(
                None,
                {
                    input_name: model_tile,
                },
            )[0]
            if stage_metrics is not None:
                stage_metrics["ai_inference_s"] = stage_metrics.get("ai_inference_s", 0.0) + time.perf_counter() - inference_started
            if tile.ndim == 3:
                del model_tile

            composition_started = time.perf_counter()

            # =================================================
            # VERIFY MODEL OUTPUT
            # =================================================

            if prediction.ndim != 4:
                raise RuntimeError(
                    "Invalid Real-ESRGAN output shape: "
                    f"{prediction.shape}"
                )

            source_tile_width = (
                right - left
            )

            source_tile_height = (
                bottom - top
            )

            expected_width = (
                source_tile_width
                * MODEL_SCALE
            )

            expected_height = (
                source_tile_height
                * MODEL_SCALE
            )

            predicted_height = (
                prediction.shape[2]
            )

            predicted_width = (
                prediction.shape[3]
            )

            if (
                predicted_width
                != expected_width
                or predicted_height
                != expected_height
            ):

                raise RuntimeError(
                    "Real-ESRGAN scale mismatch. "
                    f"Input="
                    f"{source_tile_width}x"
                    f"{source_tile_height}, "
                    f"Output="
                    f"{predicted_width}x"
                    f"{predicted_height}, "
                    f"Expected="
                    f"{expected_width}x"
                    f"{expected_height}"
                )

            # =================================================
            # MODEL → RGB
            # =================================================

            prediction = _model_to_image(
                prediction
            )

            # =================================================
            # REMOVE OVERLAP
            # =================================================

            (
                prediction,
                source_left,
                source_top,
                source_right,
                source_bottom,
            ) = _crop_ai_tile(
                prediction,
                left,
                top,
                right,
                bottom,
                source_width,
                source_height,
            )

            # =================================================
            # MAP SOURCE REGION → FINAL OUTPUT
            # =================================================

            (
                target_left,
                target_top,
                target_right,
                target_bottom,
            ) = _map_source_to_output(
                source_left,
                source_top,
                source_right,
                source_bottom,
                source_width,
                source_height,
                output_width,
                output_height,
            )

            target_width = (
                target_right
                - target_left
            )

            target_height = (
                target_bottom
                - target_top
            )

            if (
                target_width <= 0
                or target_height <= 0
            ):
                raise RuntimeError(
                    "Calculated tile output dimensions are invalid."
                )

            # =================================================
            # RESIZE AI TILE → FINAL TARGET REGION
            # =================================================

            prediction = resize_image_array(
                prediction,
                target_width,
                target_height,
            )

            # =================================================
            # WRITE FINAL CANVAS
            # =================================================

            if output_writer is not None:
                output_writer.write_tile(prediction, target_left, target_top)
            else:
                canvas[target_top:target_bottom, target_left:target_right, :] = prediction[:target_height, :target_width, :]

            if stage_metrics is not None:
                stage_metrics["composition_resize_s"] = stage_metrics.get("composition_resize_s", 0.0) + time.perf_counter() - composition_started
            composition_started = None

            # =================================================
            # PROGRESS
            # =================================================

            processed += 1

            if progress_callback:

                percent = int(
                    processed
                    / total_tiles
                    * 100
                )

                progress_callback(
                    min(
                        100,
                        max(
                            0,
                            percent,
                        ),
                    )
                )

        finally:

            if stage_metrics is not None and composition_started is not None:
                stage_metrics["composition_resize_s"] = stage_metrics.get("composition_resize_s", 0.0) + time.perf_counter() - composition_started

            # -------------------------------------------------
            # Release inference memory immediately.
            # -------------------------------------------------

            if prediction is not None:

                del prediction

            del tile

    return canvas
 
