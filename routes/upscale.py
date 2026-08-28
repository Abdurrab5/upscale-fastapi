from __future__ import annotations

import os
import re

from fastapi import (
    APIRouter,
    File,
    Form,
    UploadFile,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from starlette.background import BackgroundTask

from PIL import (
    Image,
    UnidentifiedImageError,
)

from config import (
    MAX_IMAGE_PIXELS,
    MAX_OUTPUT_DIMENSION,
    MAX_OUTPUT_PIXELS,
    MAX_UPLOAD_SIZE,
)

from services.job_manager import (
    DownloadNotReadyError,
    JobNotFoundError,
    QueueFullError,
    abort_download,
    claim_download,
    complete_download,
    get_job,
    submit_job,
)

from utils.image import (
    cleanup,
    create_input_path,
    create_output_path,
)

from utils.progress import (
    get_progress,
    remove_progress,
    update_progress,
)


router = APIRouter()


# ============================================================
# QUALITY
# ============================================================

QUALITY_MODES = {
    "2k": 2048,
    "4k": 4096,
}

DEFAULT_QUALITY = "4k"

ALLOWED_QUALITY_VALUES = set(
    QUALITY_MODES.keys()
)


# ============================================================
# FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


# ============================================================
# JOB ID
# ============================================================

def validate_job_id(
    job_id: str,
) -> bool:

    if not job_id:
        return False

    return bool(
        re.fullmatch(
            r"[A-Za-z0-9_-]{8,100}",
            job_id,
        )
    )


# ============================================================
# EXTENSION
# ============================================================

def validate_extension(
    filename: str,
) -> bool:

    extension = os.path.splitext(
        filename or ""
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# QUALITY
# ============================================================

def normalize_quality(
    quality: str,
) -> str:

    value = str(
        quality or DEFAULT_QUALITY
    ).strip().lower()

    aliases = {
        "2048": "2k",
        "2k-hd": "2k",
        "hd": "2k",

        "4096": "4k",
        "4k-hd": "4k",
    }

    value = aliases.get(
        value,
        value,
    )

    if value not in ALLOWED_QUALITY_VALUES:

        return DEFAULT_QUALITY

    return value


# ============================================================
# TARGET SAFETY
# ============================================================

def is_target_supported(
    width: int,
    height: int,
) -> bool:

    if width <= 0 or height <= 0:
        return False

    if width > MAX_OUTPUT_DIMENSION:
        return False

    if height > MAX_OUTPUT_DIMENSION:
        return False

    if (
        width * height
        > MAX_OUTPUT_PIXELS
    ):
        return False

    return True


# ============================================================
# QUALITY RESOLUTION
# ============================================================

def resolve_requested_quality(
    source_width: int,
    source_height: int,
    quality: str,
) -> str:

    requested = normalize_quality(
        quality
    )

    target_longest = QUALITY_MODES[
        requested
    ]

    source_longest = max(
        source_width,
        source_height,
    )

    scale = (
        target_longest
        / source_longest
    )

    target_width = max(
        2,
        int(
            round(
                source_width * scale
            )
        ),
    )

    target_height = max(
        2,
        int(
            round(
                source_height * scale
            )
        ),
    )

    if is_target_supported(
        target_width,
        target_height,
    ):

        return requested

    if requested != "4k":

        target_longest = QUALITY_MODES[
            "4k"
        ]

        scale = (
            target_longest
            / source_longest
        )

        target_width = max(
            2,
            int(
                round(
                    source_width * scale
                )
            ),
        )

        target_height = max(
            2,
            int(
                round(
                    source_height * scale
                )
            ),
        )

        if is_target_supported(
            target_width,
            target_height,
        ):

            return "4k"

    raise ValueError(
        "The requested output resolution "
        "cannot be safely generated."
    )


# ============================================================
# TARGET DIMENSIONS
# ============================================================

def calculate_target_dimensions(
    source_width: int,
    source_height: int,
    quality: str,
):

    target_longest = QUALITY_MODES[
        quality
    ]

    source_longest = max(
        source_width,
        source_height,
    )

    scale = (
        target_longest
        / source_longest
    )

    target_width = max(
        2,
        int(
            round(
                source_width * scale
            )
        ),
    )

    target_height = max(
        2,
        int(
            round(
                source_height * scale
            )
        ),
    )

    if not is_target_supported(
        target_width,
        target_height,
    ):

        raise ValueError(
            "The requested output resolution "
            "exceeds server limits."
        )

    return (
        target_width,
        target_height,
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image_file(
    path: str,
):

    try:

        with Image.open(path) as image:

            width, height = image.size

            image_format = (
                image.format or ""
            ).upper()

            if width <= 0 or height <= 0:

                raise ValueError(
                    "Invalid image dimensions."
                )

            pixels = (
                width * height
            )

            if pixels > MAX_IMAGE_PIXELS:

                raise ValueError(
                    "Image resolution is too large. "
                    "Please upload a smaller image."
                )

            image.verify()

            return (
                width,
                height,
                image_format,
            )

    except Image.DecompressionBombError:

        raise ValueError(
            "Image resolution is too large."
        )

    except UnidentifiedImageError:

        raise ValueError(
            "The uploaded file is not a valid image."
        )


# ============================================================
# PROGRESS
# ============================================================

@router.get(
    "/progress/{job_id}"
)
def progress(
    job_id: str,
):

    if not validate_job_id(
        job_id
    ):

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Invalid job ID.",
            },
        )

    progress_data = get_progress(
        job_id
    )

    job = get_job(
        job_id
    )

    if job is not None:

        progress_data = {
            **progress_data,

            "status": job.get(
                "status",
                "unknown",
            ),

            "quality": job.get(
                "quality"
            ),

            "error": job.get(
                "error"
            ),
        }

    return progress_data


# ============================================================
# RESULT PREVIEW
# ============================================================

@router.get(
    "/result/{job_id}"
)
def result(
    job_id: str,
):

    if not validate_job_id(
        job_id
    ):

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Invalid job ID.",
            },
        )

    job = get_job(
        job_id
    )

    if job is None:

        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": (
                    "This enhancement is no longer available."
                ),
            },
        )

    if job.get("status") != "completed":

        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": (
                    "Your image is still being enhanced."
                ),
                "status": job.get(
                    "status"
                ),
            },
        )

    output_path = job.get(
        "output_path"
    )

    if (
        not output_path
        or not os.path.isfile(output_path)
        or os.path.getsize(output_path) <= 0
    ):

        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": (
                    "The enhanced image is no longer available."
                ),
            },
        )

    return FileResponse(
        output_path,
        media_type="image/png",
        filename="enhanced-image.png",
    )


# ============================================================
# DOWNLOAD
# ============================================================

@router.get(
    "/download/{job_id}"
)
def download(
    job_id: str,
):

    if not validate_job_id(
        job_id
    ):

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Invalid job ID.",
            },
        )

    try:

        job = claim_download(
            job_id
        )

    except JobNotFoundError:

        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": (
                    "This enhanced image is no longer available."
                ),
            },
        )

    except DownloadNotReadyError as exc:

        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "message": str(exc),
            },
        )

    output_path = job.get(
        "output_path"
    )

    try:

        return FileResponse(
            output_path,
            media_type="image/png",
            filename="enhanced-image.png",
            background=BackgroundTask(
                complete_download,
                job_id,
            ),
        )

    except Exception:

        abort_download(
            job_id
        )

        raise


# ============================================================
# UPSCALE
# ============================================================

@router.post(
    "/upscale"
)
async def upscale(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    quality: str = Form(DEFAULT_QUALITY),
):

    input_path = None
    output_path = None

    try:

        # ====================================================
        # JOB ID
        # ====================================================

        if not validate_job_id(
            job_id
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Invalid job ID.",
                },
            )

        # ====================================================
        # QUALITY
        # ====================================================

        requested_quality = normalize_quality(
            quality
        )

        # ====================================================
        # FILE
        # ====================================================

        filename = (
            file.filename
            or "upload"
        )

        if not validate_extension(
            filename
        ):

            return JSONResponse(
                status_code=415,
                content={
                    "success": False,
                    "message": (
                        "Unsupported image format. "
                        "Use JPG, PNG, or WebP."
                    ),
                },
            )

        # ====================================================
        # MIME
        # ====================================================

        if (
            file.content_type
            not in ALLOWED_MIME_TYPES
        ):

            return JSONResponse(
                status_code=415,
                content={
                    "success": False,
                    "message": (
                        "Unsupported image type."
                    ),
                },
            )

        # ====================================================
        # PATHS
        # ====================================================

        input_path = create_input_path(
            filename
        )

        output_path = create_output_path()

        # ====================================================
        # UPLOAD
        # ====================================================

        update_progress(
            job_id,
            5,
            "Uploading image",
            status="uploading",
        )

        total_size = 0

        with open(
            input_path,
            "wb",
        ) as buffer:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                if (
                    total_size
                    > MAX_UPLOAD_SIZE
                ):

                    cleanup(
                        input_path,
                        output_path,
                    )

                    remove_progress(
                        job_id
                    )

                    return JSONResponse(
                        status_code=413,
                        content={
                            "success": False,
                            "message": (
                                "Image file is too large. "
                                "Maximum allowed size is 10 MB."
                            ),
                        },
                    )

                buffer.write(
                    chunk
                )

        await file.close()

        # ====================================================
        # VALIDATE
        # ====================================================

        update_progress(
            job_id,
            10,
            "Validating image",
            status="validating",
        )

        (
            source_width,
            source_height,
            image_format,
        ) = validate_image_file(
            input_path
        )

        # ====================================================
        # RESOLVE QUALITY
        # ====================================================

        final_quality = resolve_requested_quality(
            source_width,
            source_height,
            requested_quality,
        )

        # ====================================================
        # TARGET
        # ====================================================

        (
            target_width,
            target_height,
        ) = calculate_target_dimensions(
            source_width,
            source_height,
            final_quality,
        )

        # ====================================================
        # QUEUE
        # ====================================================

        update_progress(
            job_id,
            15,
            (
                f"Preparing "
                f"{final_quality.upper()} output "
                f"{target_width}×{target_height}"
            ),
            status="queued",
        )

        try:

            submit_job(
                job_id=job_id,
                input_path=input_path,
                output_path=output_path,
                quality=final_quality,
            )

        except QueueFullError:

            cleanup(
                input_path,
                output_path,
            )

            remove_progress(
                job_id
            )

            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": (
                        "The processing queue is currently "
                        "full. Please try again shortly."
                    ),
                },
            )

        # ====================================================
        # 202
        # ====================================================

        return JSONResponse(
            status_code=202,
            content={
                "success": True,

                "job_id": job_id,

                "status": "queued",

                "quality": final_quality,

                "source": {
                    "width": source_width,
                    "height": source_height,
                },

                "target": {
                    "width": target_width,
                    "height": target_height,
                },

                "message": (
                    "Your image is being enhanced. "
                    "Please wait while we prepare your result."
                ),
            },
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    except ValueError as exc:

        cleanup(
            input_path,
            output_path,
        )

        remove_progress(
            job_id
        )

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": str(exc),
            },
        )

    # ========================================================
    # IMAGE ERROR
    # ========================================================

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ):

        cleanup(
            input_path,
            output_path,
        )

        remove_progress(
            job_id
        )

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": (
                    "The uploaded file is not "
                    "a valid supported image."
                ),
            },
        )

    # ========================================================
    # UNEXPECTED ERROR
    # ========================================================

    except Exception as exc:

        print(
            f"[UPSCALE] Queue error: {exc!r}",
            flush=True,
        )

        cleanup(
            input_path,
            output_path,
        )

        remove_progress(
            job_id
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to start image enhancement. "
                    "Please try again."
                ),
            },
        )