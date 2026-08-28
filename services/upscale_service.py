
from __future__ import annotations

import gc
import time

import numpy as np
from PIL import Image, ImageOps

from services.inference import (
    get_engine,
)

from services.output_writer import (
    UpscaleOutputWriter,
)

from services.preprocess import (
    load_image,
)

from services.target_resolver import (
    resolve_target,
)

from utils.progress import (
    update_progress,
)


def upscale_image(
    input_path: str,
    output_path: str,
    job_id: str,
    quality: str,
):

    started = time.perf_counter()

    image = None
    writer = None

    try:

        # ====================================================
        # LOAD
        # ====================================================

        update_progress(
            job_id,
            5,
            "Loading image",
        )

        image = load_image(
            input_path
        )

        source_width, source_height = (
            image.original_size
        )

        print(
            "[UPSCALE] Source: "
            f"{source_width}x{source_height}",
            flush=True,
        )

        # ====================================================
        # RESOLUTION
        # ====================================================

        update_progress(
            job_id,
            10,
            "Calculating target resolution",
        )

        target = resolve_target(
            source_width,
            source_height,
            quality,
        )

        print(
            "[UPSCALE] Resolution plan:",
            target,
            flush=True,
        )

        print(
            "[UPSCALE] Strategy:",
            target.strategy,
            flush=True,
        )

        print(
            "[UPSCALE] AI passes:",
            target.ai_passes,
            flush=True,
        )

        print(
            "[UPSCALE] Scale:",
            f"{target.scale:.2f}x",
            flush=True,
        )

        # ====================================================
        # OUTPUT WRITER
        # ====================================================

        update_progress(
            job_id,
            15,
            (
                "Preparing "
                f"{target.quality.upper()} "
                f"{target.width}×"
                f"{target.height} output"
            ),
        )

        writer = UpscaleOutputWriter(
            width=target.width,
            height=target.height,
            output_path=output_path,
        )

        writer.create()

        # ====================================================
        # NON-AI RESIZE
        # ====================================================

        if not target.needs_ai:

            update_progress(
                job_id,
                35,
                "Resizing image",
            )

            with Image.open(
                input_path
            ) as source:

                source = ImageOps.exif_transpose(
                    source
                )

                source = source.convert(
                    "RGB"
                )

                resized = source.resize(
                    (
                        target.width,
                        target.height,
                    ),
                    Image.Resampling.LANCZOS,
                )

                array = np.asarray(
                    resized,
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

            if target.strategy == "fast_x2":

                label = "FSRCNN x2"

            elif target.strategy == "best_x4":

                label = "Real-ESRGAN x4"

            else:

                label = "AI enhancement"

            update_progress(
                job_id,
                20,
                (
                    f"{label} enhancement "
                    f"to "
                    f"{target.quality.upper()}"
                ),
            )

            engine = get_engine()

            result = engine.upscale(
                image.tensor,
                target_width=target.width,
                target_height=target.height,
                strategy=target.strategy,
                ai_passes=target.ai_passes,
                progress_callback=(
                    lambda percent:
                    update_progress(
                        job_id,
                        20
                        + int(
                            percent
                            * 0.70
                        ),
                        (
                            f"{label} "
                            f"processing "
                            f"({percent}%)"
                        ),
                    )
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
            - started
        )

        print(
            "[UPSCALE] Completed:",
            f"{elapsed:.2f}s",
            flush=True,
        )

        update_progress(
            job_id,
            100,
            f"Completed in {elapsed:.2f}s",
        )

        return output_path

    except Exception as exc:

        print(
            "[UPSCALE] FAILED:",
            repr(exc),
            flush=True,
        )

        update_progress(
            job_id,
            100,
            f"Processing failed: {exc}",
        )

        raise

    finally:

        if writer is not None:

            writer.close()

        if image is not None:

            try:
                del image.tensor
            except Exception:
                pass

            try:
                del image.alpha
            except Exception:
                pass

            try:
                del image
            except Exception:
                pass

        gc.collect()
 
