
import gc
import time

from services.preprocess import load_image
from services.inference import get_engine
from services.output_writer import (
    UpscaleOutputWriter,
)
from services.target_resolver import (
    resolve_target,
)

from utils.progress import (
    update_progress,
)


# ============================================================
# UPSCALE IMAGE
# ============================================================

def upscale_image(
    input_path: str,
    output_path: str,
    job_id: str,
    quality: str,
    target_width: int,
    target_height: int,
    output_format: str,
):
    """
    Perform Real-ESRGAN image upscaling.

    Supported quality modes:

        hd  -> 2048px longest edge
        2k  -> 2048px longest edge
        4k  -> 4096px longest edge

    HD is an alias of 2K.

    The target_width and target_height supplied by the route
    represent the FINAL output dimensions.

    Real-ESRGAN performs AI enhancement when required.
    Images that are already larger than the requested target
    are resized directly without AI processing.

    The source aspect ratio is preserved by the target resolver.
    """

    start = time.perf_counter()

    image = None
    writer = None

    try:

        # ====================================================
        # 1. LOAD IMAGE
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
        # 2. RESOLVE TARGET
        # ====================================================

        update_progress(
            job_id,
            15,
            "Calculating target resolution",
        )

        target_config = resolve_target(
            source_width,
            source_height,
            quality,
        )

        resolved_quality = target_config.quality
        resolved_width = target_config.width
        resolved_height = target_config.height

        # ====================================================
        # 3. TARGET CONSISTENCY
        # ====================================================

        quality = resolved_quality

        if (
            target_width <= 0
            or target_height <= 0
        ):
            target_width = resolved_width
            target_height = resolved_height

        # The route normally supplies dimensions produced by
        # the same resolver. If it does not, reject the mismatch
        # rather than silently producing an unexpected image.

        if (
            target_width != resolved_width
            or target_height != resolved_height
        ):
            raise ValueError(
                "Target dimensions do not match the resolved "
                "quality target."
            )

        ai_passes = int(
            getattr(
                target_config,
                "ai_passes",
                0,
            )
        )

        needs_ai = bool(
            getattr(
                target_config,
                "needs_ai",
                ai_passes > 0,
            )
        )

        strategy = getattr(
            target_config,
            "strategy",
            "ai" if needs_ai else "resize",
        )

        print(
            "RESOLUTION PLAN:",
            {
                "quality": quality,
                "source": (
                    source_width,
                    source_height,
                ),
                "target": (
                    target_width,
                    target_height,
                ),
                "scale": getattr(
                    target_config,
                    "scale",
                    None,
                ),
                "strategy": strategy,
                "ai_passes": ai_passes,
                "needs_ai": needs_ai,
                "output_format": output_format,
            },
            flush=True,
        )

        # ====================================================
        # 4. PREPARE OUTPUT
        # ====================================================

        update_progress(
            job_id,
            20,
            (
                f"Preparing "
                f"{quality.upper()} "
                f"{target_width}×{target_height} "
                f"{output_format.upper()} output"
            ),
        )

        writer = UpscaleOutputWriter(
            width=target_width,
            height=target_height,
            output_path=output_path,
            output_format=output_format,
        )

        writer.create()

        # ====================================================
        # 5. IMAGE PROCESSING
        # ====================================================

        if needs_ai and ai_passes > 0:

            update_progress(
                job_id,
                25,
                (
                    f"AI {quality.upper()} enhancement"
                ),
            )

            engine = get_engine()

            result = engine.upscale(
                image.tensor,
                target_width=target_width,
                target_height=target_height,
                ai_passes=ai_passes,
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

        else:

            # =================================================
            # NO AI REQUIRED
            # =================================================

            update_progress(
                job_id,
                25,
                "Resizing image",
            )

            engine = get_engine()

            result = engine.upscale(
                image.tensor,
                target_width=target_width,
                target_height=target_height,
                ai_passes=0,
                progress_callback=lambda percent:
                    update_progress(
                        job_id,
                        25 + int(
                            percent * 0.65
                        ),
                        (
                            f"Resizing "
                            f"({percent}%)"
                        ),
                    ),
            )

        # ====================================================
        # 6. VALIDATE RESULT
        # ====================================================

        if result is None:
            raise RuntimeError(
                "Image processing returned no result."
            )

        expected_shape = (
            target_height,
            target_width,
            3,
        )

        if result.shape != expected_shape:
            raise RuntimeError(
                "Unexpected processing result dimensions: "
                f"{result.shape}; expected "
                f"{expected_shape}."
            )

        if result.dtype.name != "uint8":
            raise RuntimeError(
                "Unexpected processing result dtype: "
                f"{result.dtype}."
            )

        if result.min() < 0 or result.max() > 255:
            raise RuntimeError(
                "Processing result contains invalid "
                "pixel values."
            )

        # ====================================================
        # 7. WRITE RESULT
        # ====================================================

        writer.write_tile(
            result,
            0,
            0,
        )

        del result

        # ====================================================
        # 8. FINALIZE
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

        # ====================================================
        # 9. COMPLETE
        # ====================================================

        elapsed = (
            time.perf_counter()
            - start
        )

        update_progress(
            job_id,
            100,
            (
                f"{quality.upper()} enhancement "
                f"completed in {elapsed:.2f}s"
            ),
            "completed",
        )

        return output_path

    except Exception as exc:

        # ====================================================
        # MARK FAILED
        # ====================================================

        try:

            update_progress(
                job_id,
                0,
                str(exc),
                "failed",
            )

        except Exception:
            pass

        raise

    finally:

        # ====================================================
        # CLEANUP WRITER
        # ====================================================

        if writer is not None:

            try:
                writer.close()
            except Exception:
                pass

        # ====================================================
        # CLEANUP IMAGE
        # ====================================================

        if image is not None:

            del image

        gc.collect()
 
