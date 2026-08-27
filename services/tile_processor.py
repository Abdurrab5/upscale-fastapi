import numpy as np

from config import (
    SCALE,
    TILE_SMALL,
    TILE_MEDIUM,
    TILE_LARGE,
    TILE_PAD,
)


def choose_tile_size(width: int, height: int) -> int:
    """
    Select a conservative tile size based on source resolution.

    Larger images use smaller tiles to keep RAM usage predictable.
    """

    pixels = width * height

    if pixels <= 2_000_000:
        return TILE_SMALL

    if pixels <= 6_000_000:
        return TILE_MEDIUM

    return TILE_LARGE


def generate_tiles(image):
    """
    Lazily generate overlapping tiles.

    Input:
        image: NCHW float32 NumPy array

    Yields:
        tile,
        left,
        top,
        right,
        bottom
    """

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


def _crop_overlap(
    prediction,
    left,
    top,
    right,
    bottom,
    image_width,
    image_height,
):
    """
    Remove overlapping regions from interior tile edges.

    This prevents tiles from overwriting each other and reduces
    visible seams.
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
        if right < image_width
        else 0
    )

    crop_bottom = (
        TILE_PAD
        if bottom < image_height
        else 0
    )

    height = prediction.shape[0]
    width = prediction.shape[1]

    x1 = crop_left * SCALE
    y1 = crop_top * SCALE

    x2 = width - (
        crop_right * SCALE
    )

    y2 = height - (
        crop_bottom * SCALE
    )

    cropped = prediction[
        y1:y2,
        x1:x2,
        :,
    ]

    output_left = (
        left + crop_left
    ) * SCALE

    output_top = (
        top + crop_top
    ) * SCALE

    return (
        cropped,
        output_left,
        output_top,
    )


def merge_tiles(
    session,
    image,
    tile_callback,
    progress_callback=None,
):
    """
    Run tiled ONNX inference.

    Only ONE prediction is retained at a time.

    tile_callback receives:

        tile_array
        output_left
        output_top

    The callback is responsible for writing the tile to disk.

    Returns:

        output_width,
        output_height
    """

    _, _, height, width = image.shape

    input_name = session.get_inputs()[0].name

    tile_size = choose_tile_size(
        width,
        height,
    )

    overlap = TILE_PAD * 2

    stride = max(
        tile_size - overlap,
        1,
    )

    columns = (
        max(
            0,
            width - tile_size,
        )
        + stride
        - 1
    ) // stride + 1

    rows = (
        max(
            0,
            height - tile_size,
        )
        + stride
        - 1
    ) // stride + 1

    total_tiles = max(
        rows * columns,
        1,
    )

    processed = 0

    for (
        tile,
        left,
        top,
        right,
        bottom,
    ) in generate_tiles(image):

        prediction = session.run(
            None,
            {
                input_name: tile,
            },
        )[0]

        # Remove batch dimension.
        prediction = prediction.squeeze(0)

        # CHW -> HWC.
        prediction = prediction.transpose(
            1,
            2,
            0,
        )

        # Keep values in valid image range.
        np.clip(
            prediction,
            0.0,
            1.0,
            out=prediction,
        )

        # Convert immediately to compact uint8.
        prediction = (
            prediction * 255.0
        ).astype(
            np.uint8,
        )

        (
            cropped,
            output_left,
            output_top,
        ) = _crop_overlap(
            prediction,
            left,
            top,
            right,
            bottom,
            width,
            height,
        )

        # Immediately write/process this tile.
        tile_callback(
            cropped,
            output_left,
            output_top,
        )

        processed += 1

        if progress_callback:

            percent = min(
                100,
                int(
                    processed
                    / total_tiles
                    * 100
                ),
            )

            progress_callback(
                percent
            )

        # Explicitly release large arrays.
        del prediction
        del cropped

    return (
        width * SCALE,
        height * SCALE,
    )