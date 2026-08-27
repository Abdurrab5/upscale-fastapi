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

    with Image.open(path) as source:

        img = ImageOps.exif_transpose(
            source
        )

        alpha = None

        if "A" in img.getbands():

            alpha = np.asarray(
                img.getchannel("A"),
                dtype=np.uint8,
            ).copy()

        rgb = img.convert(
            "RGB"
        )

        width, height = rgb.size

        array = np.asarray(
            rgb,
            dtype=np.uint8,
        )

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

        rgb.close()

        return ImageData(
            tensor=tensor,
            alpha=alpha,
            original_size=(
                width,
                height,
            ),
        )