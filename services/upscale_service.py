import gc
import time

import numpy as np

from PIL import Image

from services.preprocess import load_image
from services.inference import get_engine
from services.output_writer import (
    UpscaleOutputWriter,
)
from services.target_resolver import (
    resolve_target,
)

from utils.progress import update_progress


def upscale_image(
    input_path: str,
    output_path: str,
    job_id: str,
    quality: str,
):

    start = time.perf_counter()

    image = None
    writer = None

    try:

        # ====================================================
        # LOAD
        # ====================================================

        update_progress(
            job_id,
            10,
            "Loading image",
        )

        image = load_image(
            input_path
        )

        source_width, source_height = (
            image.original_size
        )

        # ====================================================
        # RESOLUTION
        # ====================================================

        update_progress(
            job_id,
            15,
            "Calculating target resolution",
        )

        target = resolve_target(
            source_width,
            source_height,
            quality,
        )

        print(
            "RESOLUTION PLAN:",
            target,
            flush=True,
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        update_progress(
            job_id,
            20,
            (
                f"Preparing {target.quality.upper()} "
                f"{target.width}×{target.height} output"
            ),
        )

        writer = UpscaleOutputWriter(
            width=target.width,
            height=target.height,
            output_path=output_path,
        )

        writer.create()

        # ====================================================
        # RESIZE ONLY
        # ====================================================

        if not target.needs_ai:

            update_progress(
                job_id,
                40,
                "Preparing high-quality output",
            )

            with Image.open(
                input_path
            ) as source:

                source = source.convert(
                    "RGB"
                )

                source = source.resize(
                    (
                        target.width,
                        target.height,
                    ),
                    Image.Resampling.LANCZOS,
                )

                array = np.asarray(
                    source,
                    dtype=np.uint8,
                ).copy()

            writer.write_tile(
                array,
                0,
                0,
            )

            del array

        # ====================================================
        # AI
        # ====================================================

        else:

            update_progress(
                job_id,
                25,
                (
                    f"AI {target.quality.upper()} "
                    f"enhancement — "
                    f"{target.strategy.replace('_', ' ')}"
                ),
            )

            engine = get_engine()

            result = engine.upscale(
                image.tensor,
                target_width=target.width,
                target_height=target.height,
                ai_passes=target.ai_passes,
                progress_callback=lambda percent:
                    update_progress(
                        job_id,
                        25 + int(
                            percent * 0.65
                        ),
                        (
                            f"AI processing "
                            f"({percent}%)"
                        ),
                    ),
            )

            writer.write_tile(
                result,
                0,
                0,
            )

            del result

        # ====================================================
        # FINALIZE
        # ====================================================

        update_progress(
            job_id,
            92,
            "Preparing final image",
        )

        writer.flush()

        update_progress(
            job_id,
            96,
            "Encoding image",
        )

        writer.finalize(
            alpha=image.alpha,
        )

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