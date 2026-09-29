
import gc
import logging
import time

from services.preprocess import load_image
from services.inference import get_engine
from services.output_writer import (
    UpscaleOutputWriter,
)
from services.target_resolver import (
    resolve_target,
)
from services.tile_processor import resize_image_array, tile_count, choose_tile_size

from utils.progress import (
    update_progress,
)
from utils.metrics import peak_rss_mb, rss_mb

# Uvicorn configures this logger at INFO and sends it to stderr, which is
# collected by local Uvicorn and container log systems.
logger = logging.getLogger("uvicorn.error")


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
    timings = {"preprocess": 0.0, "target_resolution": 0.0, "session_init": 0.0, "inference": 0.0, "ai_inference": 0.0, "composition_resize": 0.0, "encode": 0.0}

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

        stage_start = time.perf_counter()
        image = load_image(
            input_path
        )
        timings["preprocess"] = time.perf_counter() - stage_start

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

        stage_start = time.perf_counter()
        target_config = resolve_target(
            source_width,
            source_height,
            quality,
        )

        resolved_quality = target_config.quality
        resolved_width = target_config.width
        resolved_height = target_config.height
        timings["target_resolution"] = time.perf_counter() - stage_start

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

        tile = choose_tile_size(source_width, source_height)
        tiles = tile_count(source_width, source_height) if needs_ai else 0

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

            stage_start = time.perf_counter()
            engine = get_engine()
            timings["session_init"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            pipeline_timings = {"ai_inference_s": 0.0, "composition_resize_s": 0.0}
            result = engine.upscale(
                image.tensor,
                target_width=target_width,
                target_height=target_height,
                ai_passes=ai_passes,
                output_writer=writer,
                stage_metrics=pipeline_timings,
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
            timings["inference"] = time.perf_counter() - stage_start
            timings["ai_inference"] = pipeline_timings["ai_inference_s"]
            timings["composition_resize"] = pipeline_timings["composition_resize_s"]

        else:

            # =================================================
            # NO AI REQUIRED
            # =================================================

            update_progress(
                job_id,
                25,
                "Resizing image",
            )

            stage_start = time.perf_counter()
            result = resize_image_array(image.tensor, target_width, target_height)
            writer.write_tile(result, 0, 0)
            timings["inference"] = time.perf_counter() - stage_start
            timings["composition_resize"] = timings["inference"]
            del result
            update_progress(job_id, 90, "Resizing (100%)")

        # ====================================================
        # 6. VALIDATE RESULT
        # ====================================================

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

        stage_start = time.perf_counter()
        writer.finalize(
            alpha=image.alpha,
        )
        timings["encode"] = time.perf_counter() - stage_start

        # ====================================================
        # 9. COMPLETE
        # ====================================================

        elapsed = (
            time.perf_counter()
            - start
        )

        current_rss = rss_mb()
        peak_rss = peak_rss_mb()
        logger.info(
            "UPSCALE_METRICS request_id=%s source=%dx%d target=%dx%d quality=%s tile=%d tiles=%d preprocess_s=%.3f target_resolution_s=%.3f session_init_s=%.3f inference_s=%.3f ai_inference_s=%.3f composition_resize_s=%.3f encode_s=%.3f total_s=%.3f rss_mb=%s peak_rss_mb=%s",
            job_id, source_width, source_height, target_width, target_height,
            quality, tile, tiles, timings["preprocess"], timings["target_resolution"], timings["session_init"],
            timings["inference"], timings["ai_inference"], timings["composition_resize"], timings["encode"], elapsed,
            f"{current_rss:.1f}" if current_rss is not None else "unavailable",
            f"{peak_rss:.1f}" if peak_rss is not None else "unavailable",
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
 
