
import gc

import numpy as np

from PIL import (
    Image,
    ImageOps,
)


# ============================================================
# IMAGE DATA
# ============================================================

class ImageData:

    def __init__(
        self,
        tensor,
        alpha,
        original_size,
    ):
        """
        Container for the prepared source image.

        tensor:
            NCHW float32 RGB tensor in range 0.0 - 1.0.

        alpha:
            Optional uint8 alpha channel.

        original_size:
            (width, height) after EXIF orientation correction.
        """

        self.tensor = tensor
        self.alpha = alpha
        self.original_size = original_size


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    path: str,
) -> ImageData:
    """
    Load and prepare an image for Real-ESRGAN.

    Pipeline:

        File
          ↓
        EXIF orientation correction
          ↓
        Extract alpha channel
          ↓
        Convert to RGB
          ↓
        uint8 HWC
          ↓
        float32 NCHW
          ↓
        Real-ESRGAN

    The AI model receives RGB only.

    Transparency is preserved separately and can be
    restored after AI processing.

    Supported source formats:

        JPEG
        PNG
        WebP
    """

    if not path:
        raise ValueError(
            "Image path is required."
        )

    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        with Image.open(path) as source:

            # =================================================
            # EXIF ORIENTATION
            # =================================================

            img = ImageOps.exif_transpose(
                source
            )

            try:

                # =============================================
                # IMAGE SIZE
                # =============================================

                width, height = img.size

                if (
                    width <= 0
                    or height <= 0
                ):
                    raise ValueError(
                        "Invalid image dimensions."
                    )

                original_size = (
                    int(width),
                    int(height),
                )

                # =============================================
                # ALPHA CHANNEL
                # =============================================

                alpha = None

                if "A" in img.getbands():

                    alpha = np.asarray(
                        img.getchannel("A"),
                        dtype=np.uint8,
                    ).copy()

                # =============================================
                # RGB
                # =============================================
                #
                # Real-ESRGAN receives RGB only.
                #
                # PNG/WebP transparency is kept separately.
                #

                rgb = img.convert(
                    "RGB"
                )

                try:

                    array = np.asarray(
                        rgb,
                        dtype=np.uint8,
                    ).copy()

                finally:

                    rgb.close()

                # =============================================
                # VALIDATE RGB ARRAY
                # =============================================

                if array.ndim != 3:

                    raise ValueError(
                        "Unable to prepare image RGB data."
                    )

                if array.shape[2] != 3:

                    raise ValueError(
                        "Image must contain three RGB channels."
                    )

                # =============================================
                # RGB → NCHW FLOAT32
                # =============================================

                tensor = array.transpose(
                    2,
                    0,
                    1,
                )

                # Make the tensor contiguous before converting
                # to float32. This is useful for ONNX Runtime and
                # avoids unnecessary copies later.

                tensor = np.ascontiguousarray(
                    tensor,
                    dtype=np.float32,
                )

                # =============================================
                # NORMALIZE
                # =============================================

                tensor *= (
                    1.0 / 255.0
                )

                # =============================================
                # ADD BATCH DIMENSION
                # =============================================

                tensor = np.expand_dims(
                    tensor,
                    axis=0,
                )

                tensor = np.ascontiguousarray(
                    tensor,
                    dtype=np.float32,
                )

                # =============================================
                # RETURN
                # =============================================

                return ImageData(
                    tensor=tensor,
                    alpha=alpha,
                    original_size=original_size,
                )

            finally:

                # exif_transpose() can return a new image.
                # Close it when it is different from the source.
                if img is not source:

                    try:
                        img.close()
                    except Exception:
                        pass

    except (
        Image.UnidentifiedImageError,
        Image.DecompressionBombError,
    ):

        raise ValueError(
            "The uploaded file is not a valid supported image."
        )

    except OSError as exc:

        raise ValueError(
            f"Unable to read image: {exc}"
        )

    finally:

        gc.collect()
 
