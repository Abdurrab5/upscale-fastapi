import numpy as np

from PIL import Image

from config import (
    ESRGAN_TILE_LARGE,
    ESRGAN_TILE_MEDIUM,
    ESRGAN_TILE_SMALL,
    FAST_TILE_LARGE,
    FAST_TILE_MEDIUM,
    FAST_TILE_SMALL,
    TILE_PAD,
)


# ============================================================
# TILE SIZE
# ============================================================

def choose_tile_size(
    width: int,
    height: int,
    model_scale: int,
) -> int:

    pixels = (
        width
        * height
    )

    if model_scale == 2:

        if pixels <= 2_000_000:
            return FAST_TILE_SMALL

        if pixels <= 6_000_000:
            return FAST_TILE_MEDIUM

        return FAST_TILE_LARGE

    if pixels <= 2_000_000:
        return ESRGAN_TILE_SMALL

    if pixels <= 6_000_000:
        return ESRGAN_TILE_MEDIUM

    return ESRGAN_TILE_LARGE


# ============================================================
# TILE GENERATOR
# ============================================================

def generate_tiles(
    image,
    model_scale: int,
):

    _, _, height, width = (
        image.shape
    )

    tile_size = choose_tile_size(
        width,
        height,
        model_scale,
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

            yield (
                image[
                    :,
                    :,
                    top:bottom,
                    left:right,
                ],
                left,
                top,
                right,
                bottom,
            )


# ============================================================
# OUTPUT CONVERSION
# ============================================================

def model_output_to_image(
    prediction,
):

    if prediction.ndim != 4:

        raise RuntimeError(
            "Unexpected ONNX output shape: "
            f"{prediction.shape}"
        )

    prediction = prediction[0]

    if prediction.shape[0] == 3:

        prediction = prediction.transpose(
            1,
            2,
            0,
        )

    elif prediction.shape[-1] == 3:

        pass

    else:

        raise RuntimeError(
            "Model output must contain 3 RGB channels."
        )

    prediction = np.clip(
        prediction,
        0.0,
        1.0,
    )

    prediction = (
        prediction
        * 255.0
    ).astype(
        np.uint8
    )

    return np.ascontiguousarray(
        prediction
    )


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

        resized = image.resize(
            (
                int(width),
                int(height),
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
# AI TILE PASS
# ============================================================

def run_ai_pass(
    session,
    image,
    output_width,
    output_height,
    model_scale,
    progress_callback=None,
):

    (
        _,
        _,
        source_height,
        source_width,
    ) = image.shape

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    # --------------------------------------------------------
    # Final output canvas
    # --------------------------------------------------------

    canvas = np.zeros(
        (
            output_height,
            output_width,
            3,
        ),
        dtype=np.uint8,
    )

    tile_size = choose_tile_size(
        source_width,
        source_height,
        model_scale,
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

    processed = 0

    for (
        tile,
        left,
        top,
        right,
        bottom,
    ) in generate_tiles(
        image,
        model_scale,
    ):

        tile = np.ascontiguousarray(
            tile,
            dtype=np.float32,
        )

        prediction = session.run(
            None,
            {
                input_name: tile,
            },
        )[0]

        prediction = (
            model_output_to_image(
                prediction
            )
        )

        # ----------------------------------------------------
        # Verify model scale
        # ----------------------------------------------------

        expected_width = (
            tile.shape[3]
            * model_scale
        )

        expected_height = (
            tile.shape[2]
            * model_scale
        )

        if (
            prediction.shape[1]
            != expected_width
            or prediction.shape[0]
            != expected_height
        ):

            raise RuntimeError(
                "Model scale mismatch. "
                f"Input={tile.shape[3]}x"
                f"{tile.shape[2]}, "
                f"Output={prediction.shape[1]}x"
                f"{prediction.shape[0]}, "
                f"Expected={expected_width}x"
                f"{expected_height}"
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
            * model_scale
        )

        y1 = (
            crop_top
            * model_scale
        )

        x2 = (
            prediction.shape[1]
            - crop_right
            * model_scale
        )

        y2 = (
            prediction.shape[0]
            - crop_bottom
            * model_scale
        )

        prediction = prediction[
            y1:y2,
            x1:x2,
        ]

        # ----------------------------------------------------
        # Source coordinates
        # ----------------------------------------------------

        source_left = (
            left
            + crop_left
        )

        source_top = (
            top
            + crop_top
        )

        source_right = (
            right
            - crop_right
        )

        source_bottom = (
            bottom
            - crop_bottom
        )

        # ----------------------------------------------------
        # Map source to final output
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
            target_right
            - target_left,
        )

        target_height = max(
            1,
            target_bottom
            - target_top,
        )

        prediction = resize_image_array(
            prediction,
            target_width,
            target_height,
        )

        # ----------------------------------------------------
        # Safe canvas write
        # ----------------------------------------------------

        right_limit = min(
            output_width,
            target_left
            + prediction.shape[1],
        )

        bottom_limit = min(
            output_height,
            target_top
            + prediction.shape[0],
        )

        write_width = (
            right_limit
            - target_left
        )

        write_height = (
            bottom_limit
            - target_top
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

        del tile
        del prediction

    return canvas