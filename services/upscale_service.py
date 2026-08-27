import gc
import time

from config import SCALE

from services.preprocess import load_image
from services.inference import get_engine
from services.output_writer import UpscaleOutputWriter

from utils.progress import update_progress


def upscale_image(
    input_path: str,
    output_path: str,
    job_id: str,
):
    """
    CPU-bound production upscaling pipeline.

    This function intentionally remains synchronous.
    The FastAPI route executes it in a worker thread.
    """

    start = time.perf_counter()

    image = None
    writer = None

    try:

        # ========================================================
        # LOAD IMAGE
        # ========================================================

        update_progress(
            job_id,
            10,
            "Loading image",
        )

        image = load_image(
            input_path
        )

        # ========================================================
        # LOAD MODEL
        # ========================================================

        update_progress(
            job_id,
            20,
            "Loading AI model",
        )

        engine = get_engine()

        # ========================================================
        # OUTPUT DIMENSIONS
        # ========================================================

        original_width, original_height = (
            image.original_size
        )

        output_width = (
            original_width * SCALE
        )

        output_height = (
            original_height * SCALE
        )

        # ========================================================
        # OUTPUT WRITER
        # ========================================================

        update_progress(
            job_id,
            25,
            "Preparing output",
        )

        writer = UpscaleOutputWriter(
            width=output_width,
            height=output_height,
            output_path=output_path,
        )

        writer.create()

        # ========================================================
        # TILE CALLBACK
        # ========================================================

        def write_tile(
            tile,
            left,
            top,
        ):

            writer.write_tile(
                tile,
                left,
                top,
            )

        # ========================================================
        # INFERENCE
        # ========================================================

        update_progress(
            job_id,
            30,
            "Upscaling image",
        )

        engine.upscale(
            image.tensor,
            tile_callback=write_tile,
            progress_callback=lambda percent:
                update_progress(
                    job_id,
                    30 + int(
                        percent * 0.60
                    ),
                    f"Upscaling ({percent}%)",
                ),
        )

        # ========================================================
        # FLUSH
        # ========================================================

        update_progress(
            job_id,
            92,
            "Preparing output",
        )

        writer.flush()

        # ========================================================
        # PNG
        # ========================================================

        update_progress(
            job_id,
            95,
            "Encoding image",
        )

        writer.finalize(
            alpha=image.alpha,
        )

        # ========================================================
        # COMPLETE
        # ========================================================

        elapsed = (
            time.perf_counter()
            - start
        )

        update_progress(
            job_id,
            100,
            f"Completed in {elapsed:.2f}s",
        )

        return output_path

    finally:

        if writer is not None:
            writer.close()

        if image is not None:
            del image

        gc.collect()