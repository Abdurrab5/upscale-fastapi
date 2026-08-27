import gc

import numpy as np

from PIL import (
    Image,
    ImageOps,
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


def load_image(
    path: str,
) -> ImageData:
    """
    Load and prepare an image for ONNX inference.

    Pipeline:

        file
          ↓
        EXIF correction
          ↓
        RGB
          ↓
        uint8 NumPy
          ↓
        float32 NCHW tensor
    """

    with Image.open(path) as source:

        # -----------------------------------------------------
        # EXIF ORIENTATION
        # -----------------------------------------------------

        img = ImageOps.exif_transpose(
            source
        )

        # -----------------------------------------------------
        # ALPHA
        # -----------------------------------------------------

        alpha = None

        if "A" in img.getbands():

            alpha = np.asarray(
                img.getchannel("A"),
                dtype=np.uint8,
            ).copy()

        # -----------------------------------------------------
        # RGB
        # -----------------------------------------------------

        rgb = img.convert(
            "RGB"
        )

        width, height = rgb.size

        # -----------------------------------------------------
        # UINT8 ARRAY
        # -----------------------------------------------------

        array = np.asarray(
            rgb,
            dtype=np.uint8,
        )

        # -----------------------------------------------------
        # FLOAT32 NCHW
        # -----------------------------------------------------

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

        # Normalize in-place.
        tensor *= (
            1.0 / 255.0
        )

        # Add batch dimension.
        tensor = np.expand_dims(
            tensor,
            axis=0,
        )

        # Release temporary PIL/array objects.
        rgb.close()

        del array

        gc.collect()

        return ImageData(
            tensor=tensor,
            alpha=alpha,
            original_size=(
                width,
                height,
            ),
        )