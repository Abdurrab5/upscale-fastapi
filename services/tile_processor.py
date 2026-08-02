import numpy as np

from config import (
    SCALE,
    TILE_SMALL,
    TILE_MEDIUM,
    TILE_LARGE,
    TILE_PAD
)


def choose_tile_size(width, height):
    """
    Choose tile size based on image resolution.
    """

    pixels = width * height

    if pixels <= 2_000_000:
        return TILE_SMALL

    if pixels <= 8_000_000:
        return TILE_MEDIUM

    return TILE_LARGE


def generate_tiles(image):
    """
    Generate image tiles.

    Input:
        image -> (1,3,H,W)

    Returns:
        tile,
        left,
        top,
        right,
        bottom
    """

    _, _, h, w = image.shape

    tile_size = choose_tile_size(w, h)

    stride = tile_size - TILE_PAD * 2

    for top in range(0, h, stride):

        for left in range(0, w, stride):

            right = min(left + tile_size, w)
            bottom = min(top + tile_size, h)

            tile = image[
                :,
                :,
                top:bottom,
                left:right
            ]

            yield (
                tile,
                left,
                top,
                right,
                bottom
            )


def merge_tiles(session, image, progress_callback=None):
    """
    Run tiled inference and merge results.

    Parameters
    ----------
    session:
        ONNX Runtime session

    image:
        numpy array
        shape = (1,3,H,W)

    Returns
    -------
    output tensor
    """

    _, _, h, w = image.shape

    output = np.zeros(
        (
            1,
            3,
            h * SCALE,
            w * SCALE
        ),
        dtype=np.float32
    )

    input_name = session.get_inputs()[0].name

    tiles = list(generate_tiles(image))

    total = len(tiles)

    for index, (
        tile,
        left,
        top,
        right,
        bottom
    ) in enumerate(tiles):

        prediction = session.run(
            None,
            {
                input_name: tile
            }
        )[0]

        output[
            :,
            :,
            top * SCALE: bottom * SCALE,
            left * SCALE: right * SCALE
        ] = prediction

        if progress_callback:

            percent = int(
                ((index + 1) / total) * 100
            )

            progress_callback(percent)

    return output