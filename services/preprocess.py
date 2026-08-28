import numpy as np

from PIL import (
    Image,
    ImageOps,
)

from config import (
    MAX_IMAGE_PIXELS,
)


class ImageData:

    def __init__(
        self,
        tensor,
        alpha,
        original_size,
    ):

        self.tensor = tensor
        self.alpha = alpha
        self.original_size = original_size


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    path: str,
) -> ImageData:

    with Image.open(path) as source:

        source = ImageOps.exif_transpose(
            source
        )

        width, height = source.size

        pixels = (
            width
            * height
        )

        if pixels > MAX_IMAGE_PIXELS:

            raise ValueError(
                "Image exceeds the maximum "
                f"allowed resolution of "
                f"{MAX_IMAGE_PIXELS:,} pixels."
            )

        # ----------------------------------------------------
        # Alpha
        # ----------------------------------------------------

        alpha = None

        if "A" in source.getbands():

            alpha = np.asarray(
                source.getchannel("A"),
                dtype=np.uint8,
            ).copy()

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb = source.convert(
            "RGB"
        )

        try:

            array = np.asarray(
                rgb,
                dtype=np.uint8,
            ).copy()

        finally:

            rgb.close()

        # ----------------------------------------------------
        # RGB HWC -> NCHW
        # ----------------------------------------------------

        tensor = (
            array
            .transpose(
                2,
                0,
                1,
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        tensor *= (
            1.0 / 255.0
        )

        tensor = np.expand_dims(
            tensor,
            axis=0,
        )

        del array

        return ImageData(
            tensor=tensor,
            alpha=alpha,
            original_size=(
                width,
                height,
            ),
        )