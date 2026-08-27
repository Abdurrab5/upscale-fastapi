import gc
import os
import tempfile

import numpy as np
from PIL import Image


class UpscaleOutputWriter:
    """
    Disk-backed RGB output canvas.

    The complete upscaled RGB image is stored in a memory-mapped
    file rather than a normal NumPy array.
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

    # =========================================================
    # CREATE
    # =========================================================

    def create(self):
        """
        Create the disk-backed RGB canvas.
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

        shape = (
            self.height,
            self.width,
            3,
        )

        self.canvas = np.memmap(
            self.temp_path,
            dtype=np.uint8,
            mode="w+",
            shape=shape,
        )

        # Initialize output to black.
        self.canvas[:] = 0

        self.canvas.flush()

    # =========================================================
    # WRITE TILE
    # =========================================================

    def write_tile(
        self,
        tile,
        left: int,
        top: int,
    ):
        """
        Write one processed tile into the output canvas.
        """

        if self.canvas is None:
            raise RuntimeError(
                "Output writer has not been initialized."
            )

        if tile is None:
            return

        if tile.ndim != 3:
            raise ValueError(
                "Output tile must have shape HWC."
            )

        tile_height = tile.shape[0]
        tile_width = tile.shape[1]

        if tile_height <= 0 or tile_width <= 0:
            return

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
            left + tile_width,
        )

        bottom = min(
            self.height,
            top + tile_height,
        )

        if right <= left or bottom <= top:
            return

        write_width = right - left
        write_height = bottom - top

        self.canvas[
            top:bottom,
            left:right,
            :
        ] = tile[
            :write_height,
            :write_width,
            :
        ]

    # =========================================================
    # FLUSH
    # =========================================================

    def flush(self):
        """
        Flush pending memory-mapped writes to disk.
        """

        if self.canvas is not None:
            self.canvas.flush()

    # =========================================================
    # FINALIZE
    # =========================================================

    def finalize(
        self,
        alpha=None,
    ):
        """
        Encode the completed output as PNG.

        The RGB canvas remains disk-backed until PIL starts
        encoding the final image.
        """

        if self.canvas is None:
            raise RuntimeError(
                "Output writer has not been initialized."
            )

        self.flush()

        # PIL can consume the memmap without us explicitly
        # creating another NumPy copy here.
        image = Image.fromarray(
            self.canvas,
            mode="RGB",
        )

        alpha_image = None

        try:

            if alpha is not None:

                alpha_image = Image.fromarray(
                    alpha,
                    mode="L",
                )

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

            image.save(
                self.output_path,
                format="PNG",
                compress_level=6,
            )

        finally:

            if alpha_image is not None:
                alpha_image.close()

            image.close()

            gc.collect()

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):
        """
        Release the mmap and remove the raw temporary file.
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

    # =========================================================
    # CONTEXT MANAGER
    # =========================================================

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