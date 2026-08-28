import gc
import os
import tempfile

import numpy as np

from PIL import Image

from config import (
    OUTPUT_FORMAT,
    PNG_COMPRESS_LEVEL,
)


class UpscaleOutputWriter:

    """
    Disk-backed RGB output canvas.

    The canvas is always the FINAL requested
    resolution.
    """

    def __init__(
        self,
        width: int,
        height: int,
        output_path: str,
    ):

        self.width = int(width)
        self.height = int(height)

        self.output_path = output_path

        self.temp_path = None
        self.canvas = None


    # ========================================================
    # CREATE
    # ========================================================

    def create(self):

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

        fd, self.temp_path = (
            tempfile.mkstemp(
                prefix="upscale_",
                suffix=".raw",
                dir=directory,
            )
        )

        os.close(fd)

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

        self.canvas[:] = 0


    # ========================================================
    # WRITE
    # ========================================================

    def write_tile(
        self,
        tile,
        left,
        top,
    ):

        if self.canvas is None:

            raise RuntimeError(
                "Output writer has not been initialized."
            )

        if tile is None:
            return

        tile = np.asarray(
            tile
        )

        if tile.ndim != 3:

            raise ValueError(
                "Output tile must be HWC."
            )

        left = max(
            0,
            int(left),
        )

        top = max(
            0,
            int(top),
        )

        right = min(
            self.width,
            left + tile.shape[1],
        )

        bottom = min(
            self.height,
            top + tile.shape[0],
        )

        if (
            right <= left
            or bottom <= top
        ):
            return

        self.canvas[
            top:bottom,
            left:right,
        ] = tile[
            :bottom - top,
            :right - left,
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

        if self.canvas is None:

            raise RuntimeError(
                "Output writer has not been initialized."
            )

        self.flush()

        # Copy the memmap into a regular ndarray
        # before handing it to Pillow.

        rgb = np.asarray(
            self.canvas
        ).copy()

        image = Image.fromarray(
            rgb,
            mode="RGB",
        )

        del rgb

        alpha_image = None

        try:

            if alpha is not None:

                alpha_image = Image.fromarray(
                    alpha,
                    mode="L",
                )

                if alpha_image.size != image.size:

                    resized = alpha_image.resize(
                        image.size,
                        Image.Resampling.LANCZOS,
                    )

                    alpha_image.close()

                    alpha_image = resized

                image.putalpha(
                    alpha_image
                )

            if OUTPUT_FORMAT.upper() == "PNG":

                image.save(
                    self.output_path,
                    format="PNG",
                    compress_level=PNG_COMPRESS_LEVEL,
                )

            else:

                image.save(
                    self.output_path,
                    format=OUTPUT_FORMAT,
                )

        finally:

            if alpha_image is not None:
                alpha_image.close()

            image.close()

            gc.collect()


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

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