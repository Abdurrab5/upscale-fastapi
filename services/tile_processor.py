import numpy as np

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

    pixels = width * height

    if pixels <= 2_000_000:
        return TILE_SMALL

    if pixels <= 6_000_000:
        return TILE_MEDIUM

    return TILE_LARGE


# ============================================================
# GENERATE TILES
# ============================================================

def generate_tiles(
    image,
):

    _, _, height, width = image.shape

    tile_size = choose_tile_size(
        width,
        height,
    )

    overlap = TILE_PAD * 2

    stride = max(
        tile_size - overlap,
        1,
    )

    for top in range(
        0,
        height,
        stride,
    ):

        for left in range(
            0,
            width,
            stride,
        ):

            right = min(
                left + tile_size,
                width,
            )

            bottom = min(
                top + tile_size,
                height,
            )

            tile = image[
                :,
                :,
                top:bottom,
                left:right,
            ]

            yield (
                tile,
                left,
                top,
                right,
                bottom,
            )


# ============================================================
# MODEL OUTPUT
# ============================================================

def _model_to_image(
    prediction,
):

    if prediction.ndim != 4:

        raise RuntimeError(
            "Unexpected ONNX model output: "
            f"{prediction.shape}"
        )

    prediction = prediction.squeeze(
        0
    )

    prediction = prediction.transpose(
        1,
        2,
        0,
    )

    np.clip(
        prediction,
        0.0,
        1.0,
        out=prediction,
    )

    prediction = (
        prediction * 255.0
    ).astype(
        np.uint8
    )

    return prediction


# ============================================================
# RESIZE
# ============================================================

def resize_image_array(
    array,
    width,
    height,
):

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

        image = image.resize(
            (
                width,
                height,
            ),
            Image.Resampling.LANCZOS,
        )

        return np.asarray(
            image,
            dtype=np.uint8,
        ).copy()

    finally:

        image.close()


# ============================================================
# SINGLE AI PASS
# ============================================================

def run_ai_pass(
    session,
    image,
    output_width,
    output_height,
    progress_callback=None,
):

    _, _, source_height, source_width = (
        image.shape
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    tile_size = choose_tile_size(
        source_width,
        source_height,
    )

    overlap = TILE_PAD * 2

    stride = max(
        tile_size - overlap,
        1,
    )

    columns = (
        max(
            0,
            source_width - tile_size,
        )
        + stride
        - 1
    ) // stride + 1

    rows = (
        max(
            0,
            source_height - tile_size,
        )
        + stride
        - 1
    ) // stride + 1

    total_tiles = max(
        1,
        rows * columns,
    )

    # --------------------------------------------------------
    # Final stage canvas
    # --------------------------------------------------------

    canvas = np.zeros(
        (
            output_height,
            output_width,
            3,
        ),
        dtype=np.uint8,
    )

    processed = 0

    for (
        tile,
        left,
        top,
        right,
        bottom,
    ) in generate_tiles(image):

        print(
            "DEBUG MODEL INPUT:",
            tile.shape,
            flush=True,
        )

        prediction = session.run(
            None,
            {
                input_name: tile,
            },
        )[0]

        print(
            "DEBUG MODEL OUTPUT:",
            prediction.shape,
            flush=True,
        )

        if prediction.ndim != 4:

            raise RuntimeError(
                "Invalid RealESRGAN output."
            )

        predicted_height = (
            prediction.shape[2]
        )

        predicted_width = (
            prediction.shape[3]
        )

        source_tile_height = (
            tile.shape[2]
        )

        source_tile_width = (
            tile.shape[3]
        )

        expected_width = (
            source_tile_width
            * MODEL_SCALE
        )

        expected_height = (
            source_tile_height
            * MODEL_SCALE
        )

        if (
            predicted_width
            != expected_width
            or predicted_height
            != expected_height
        ):

            raise RuntimeError(
                "RealESRGAN scale mismatch. "
                f"Input={source_tile_width}x"
                f"{source_tile_height}, "
                f"Output={predicted_width}x"
                f"{predicted_height}, "
                f"Expected={expected_width}x"
                f"{expected_height}"
            )

        prediction = _model_to_image(
            prediction
        )

        # ----------------------------------------------------
        # Remove overlap
        # ----------------------------------------------------

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

        x1 = (
            crop_left
            * MODEL_SCALE
        )

        y1 = (
            crop_top
            * MODEL_SCALE
        )

        x2 = (
            prediction.shape[1]
            - crop_right * MODEL_SCALE
        )

        y2 = (
            prediction.shape[0]
            - crop_bottom * MODEL_SCALE
        )

        prediction = prediction[
            y1:y2,
            x1:x2,
        ]

        # ----------------------------------------------------
        # Source coordinates after overlap
        # ----------------------------------------------------

        source_left = (
            left + crop_left
        )

        source_top = (
            top + crop_top
        )

        source_right = (
            right - crop_right
        )

        source_bottom = (
            bottom - crop_bottom
        )

        # ----------------------------------------------------
        # Map to final stage
        # ----------------------------------------------------

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

        target_width = max(
            1,
            target_right - target_left,
        )

        target_height = max(
            1,
            target_bottom - target_top,
        )

        prediction = resize_image_array(
            prediction,
            target_width,
            target_height,
        )

        right_limit = min(
            output_width,
            target_left + prediction.shape[1],
        )

        bottom_limit = min(
            output_height,
            target_top + prediction.shape[0],
        )

        write_width = (
            right_limit - target_left
        )

        write_height = (
            bottom_limit - target_top
        )

        if (
            write_width > 0
            and write_height > 0
        ):

            canvas[
                target_top:bottom_limit,
                target_left:right_limit,
            ] = prediction[
                :write_height,
                :write_width,
            ]

        processed += 1

        if progress_callback:

            progress_callback(
                min(
                    100,
                    int(
                        processed
                        / total_tiles
                        * 100
                    ),
                )
            )

        del prediction

    return canvas