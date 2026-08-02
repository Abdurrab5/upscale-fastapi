import gc
import time

from services.preprocess import load_image
from services.postprocess import tensor_to_image
from services.inference import get_engine

from utils.progress import update_progress


async def upscale_image(
    input_path: str,
    output_path: str,
    job_id: str
):
    start = time.perf_counter()

    update_progress(
        job_id,
        10,
        "Loading image"
    )

    image = load_image(input_path)

    update_progress(
        job_id,
        20,
        "Loading AI model"
    )

    engine = get_engine()

    update_progress(
        job_id,
        30,
        "Upscaling image"
    )

    output = engine.upscale(
        image.tensor,
        progress_callback=lambda p: update_progress(
            job_id,
            30 + int(p * 0.6),
            f"Upscaling ({p}%)"
        )
    )

    update_progress(
        job_id,
        92,
        "Encoding image"
    )

    result = tensor_to_image(
        output,
        image.alpha
    )

    result.save(
        output_path,
        format="PNG",
        optimize=True
    )

    del output
    del image
    del result

    gc.collect()

    elapsed = time.perf_counter() - start

    update_progress(
        job_id,
        100,
        f"Completed in {elapsed:.2f}s"
    )

    return output_path