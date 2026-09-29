
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
        rgb_array,
        alpha,
        original_size,
    ):
        """
        Container for the prepared source image.

        rgb:
            HWC uint8 RGB source. Individual inference tiles are
            converted to NCHW float32 immediately before ONNX.

        alpha:
            Optional uint8 alpha channel.

        original_size:
            (width, height) after EXIF orientation correction.
        """

        self.tensor = rgb_array
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
        HWC uint8 source, with each tile converted just in time
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

                    rgb_array = np.asarray(
                        rgb,
                        dtype=np.uint8,
                    ).copy()

                finally:

                    rgb.close()

                # =============================================
                # VALIDATE RGB ARRAY
                # =============================================

                if rgb_array.ndim != 3:

                    raise ValueError(
                        "Unable to prepare image RGB data."
                    )

                if rgb_array.shape[2] != 3:

                    raise ValueError(
                        "Image must contain three RGB channels."
                    )

                # =============================================
                # Keep the decoded image compact. Inference creates
                # float32 NCHW only for the current tile.

                # =============================================
                # RETURN
                # =============================================

                return ImageData(
                    rgb_array=rgb_array,
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
