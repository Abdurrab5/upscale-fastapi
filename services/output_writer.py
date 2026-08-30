
import gc
import os
import tempfile

import numpy as np

from PIL import Image

from config import (
    PNG_COMPRESS_LEVEL,
)


# ============================================================
# UPSCALE OUTPUT WRITER
# ============================================================

class UpscaleOutputWriter:

    """
    Disk-backed final output canvas.

    The canvas represents the FINAL requested output
    resolution.

    It does NOT store the Real-ESRGAN intermediate x4
    resolution.

    Supported output formats:

        png
        jpeg
        webp

    JPG is normalized to JPEG internally.
    """

    def __init__(
        self,
        width: int,
        height: int,
        output_path: str,
        output_format: str = "png",
    ):

        self.width = int(width)
        self.height = int(height)

        if self.width <= 0:
            raise ValueError(
                "Output width must be greater than zero."
            )

        if self.height <= 0:
            raise ValueError(
                "Output height must be greater than zero."
            )

        self.output_path = output_path

        self.output_format = (
            str(
                output_format or "png"
            )
            .strip()
            .lower()
        )

        # ----------------------------------------------------
        # Normalize JPG -> JPEG
        # ----------------------------------------------------

        if self.output_format == "jpg":
            self.output_format = "jpeg"

        # ----------------------------------------------------
        # Validate output format
        # ----------------------------------------------------

        if self.output_format not in {
            "png",
            "jpeg",
            "webp",
        }:
            raise ValueError(
                "Unsupported output format."
            )

        self.temp_path = None
        self.canvas = None

    # ========================================================
    # CREATE
    # ========================================================

    def create(self):
        """
        Create the disk-backed final RGB canvas.

        The canvas uses:

            height × width × 3

        uint8 RGB storage.
        """

        if self.canvas is not None:
            return

        directory = os.path.dirname(
            os.path.abspath(
                self.output_path
            )
        )

        os.makedirs(
            directory,
            exist_ok=True,
        )

        fd, self.temp_path = tempfile.mkstemp(
            prefix="upscale_",
            suffix=".raw",
            dir=directory,
        )

        os.close(fd)

        try:

            self.canvas = np.memmap(
                self.temp_path,
                dtype=np.uint8,
                mode="w+",
                shape=(
                    self.height,
                    self.width,
                    3,
                ),
            )

            # ------------------------------------------------
            # Initialize canvas.
            # ------------------------------------------------

            self.canvas[:] = 0

        except Exception:

            self.canvas = None

            try:

                if os.path.isfile(
                    self.temp_path
                ):
                    os.remove(
                        self.temp_path
                    )

            except OSError:
                pass

            self.temp_path = None

            raise

    # ========================================================
    # WRITE TILE
    # ========================================================

    def write_tile(
        self,
        tile,
        left: int,
        top: int,
    ):
        """
        Write an RGB/RGBA tile into the final canvas.

        Coordinates are clipped automatically to the final
        output dimensions.
        """

        if self.canvas is None:
            raise RuntimeError(
                "Output writer has not been initialized."
            )

        if tile is None:
            return

        # ----------------------------------------------------
        # Convert supported array-like input to numpy.
        # ----------------------------------------------------

        tile = np.asarray(
            tile
        )

        if tile.ndim != 3:
            raise ValueError(
                "Output tile must have shape HWC."
            )

        tile_height = int(
            tile.shape[0]
        )

        tile_width = int(
            tile.shape[1]
        )

        channels = int(
            tile.shape[2]
        )

        if (
            tile_height <= 0
            or tile_width <= 0
        ):
            return

        if channels not in {
            3,
            4,
        }:
            raise ValueError(
                "Output tile must contain "
                "3 RGB or 4 RGBA channels."
            )

        # ----------------------------------------------------
        # Normalize coordinates.
        # ----------------------------------------------------

        left = max(
            0,
            int(left),
        )

        top = max(
            0,
            int(top),
        )

        # ----------------------------------------------------
        # Clip tile to final canvas.
        # ----------------------------------------------------

        right = min(
            self.width,
            left + tile_width,
        )

        bottom = min(
            self.height,
            top + tile_height,
        )

        if (
            right <= left
            or bottom <= top
        ):
            return

        write_width = (
            right - left
        )

        write_height = (
            bottom - top
        )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        if channels == 3:

            self.canvas[
                top:bottom,
                left:right,
                :
            ] = tile[
                :write_height,
                :write_width,
                :3,
            ]

        # ----------------------------------------------------
        # RGBA
        #
        # Output canvas itself remains RGB.
        # Transparency is handled separately through the
        # alpha channel supplied to finalize().
        # ----------------------------------------------------

        else:

            self.canvas[
                top:bottom,
                left:right,
                :
            ] = tile[
                :write_height,
                :write_width,
                :3,
            ]

    # ========================================================
    # FLUSH
    # ========================================================

    def flush(self):

        if self.canvas is not None:

            self.canvas.flush()

    # ========================================================
    # FINALIZE
    # ========================================================

    def finalize(
        self,
        alpha=None,
    ):
        """
        Convert the disk-backed RGB canvas into the requested
        final image format.

        PNG:

            - preserves transparency when alpha is supplied
            - remains PNG
            - does not convert to JPEG

        JPEG:

            - always RGB
            - transparency is discarded because JPEG does
              not support alpha

        WebP:

            - RGB output
            - transparency is currently preserved only when
              the writer is explicitly extended for WebP alpha
        """

        if self.canvas is None:
            raise RuntimeError(
                "Output writer has not been initialized."
            )

        self.flush()

        image = None
        alpha_image = None

        try:

            # =================================================
            # CREATE RGB IMAGE
            # =================================================

            image = Image.fromarray(
                self.canvas,
                mode="RGB",
            )

            # =================================================
            # PNG TRANSPARENCY
            # =================================================

            if (
                alpha is not None
                and self.output_format == "png"
            ):

                alpha_array = np.asarray(
                    alpha
                )

                if alpha_array.ndim != 2:

                    raise ValueError(
                        "Alpha channel must have shape HW."
                    )

                alpha_image = Image.fromarray(
                    alpha_array.astype(
                        np.uint8,
                        copy=False,
                    ),
                    mode="L",
                )

                # ---------------------------------------------
                # Ensure alpha matches FINAL dimensions.
                # ---------------------------------------------

                if alpha_image.size != (
                    self.width,
                    self.height,
                ):

                    alpha_image = alpha_image.resize(
                        (
                            self.width,
                            self.height,
                        ),
                        Image.Resampling.LANCZOS,
                    )

                image.putalpha(
                    alpha_image
                )

            # =================================================
            # PNG
            # =================================================

            if self.output_format == "png":

                image.save(
                    self.output_path,
                    format="PNG",
                    compress_level=PNG_COMPRESS_LEVEL,
                )

            # =================================================
            # JPEG
            # =================================================

            elif self.output_format == "jpeg":

                if image.mode != "RGB":

                    image = image.convert(
                        "RGB"
                    )

                image.save(
                    self.output_path,
                    format="JPEG",
                    quality=95,
                    optimize=True,
                )

            # =================================================
            # WEBP
            # =================================================

            elif self.output_format == "webp":

                image.save(
                    self.output_path,
                    format="WEBP",
                    quality=95,
                    method=4,
                )

            # =================================================
            # VERIFY OUTPUT FILE
            # =================================================

            if (
                not os.path.isfile(
                    self.output_path
                )
                or os.path.getsize(
                    self.output_path
                ) <= 0
            ):

                raise RuntimeError(
                    "Output image was not created."
                )

        finally:

            if alpha_image is not None:

                try:
                    alpha_image.close()
                except Exception:
                    pass

            if image is not None:

                try:
                    image.close()
                except Exception:
                    pass

            gc.collect()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        """
        Release the memmap and remove its raw temporary file.

        The finalized output file is NOT removed here.

        The caller owns the final output file lifecycle.
        """

        if self.canvas is not None:

            try:
                self.canvas.flush()
            except Exception:
                pass

            try:
                del self.canvas
            except Exception:
                pass

            self.canvas = None

        gc.collect()

        # ----------------------------------------------------
        # Remove raw memmap backing file.
        # ----------------------------------------------------

        if self.temp_path is not None:

            try:

                if os.path.isfile(
                    self.temp_path
                ):
                    os.remove(
                        self.temp_path
                    )

            except OSError:
                pass

            self.temp_path = None

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(self):

        self.create()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()
 
