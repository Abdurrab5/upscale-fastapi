import asyncio
import os
import re
import traceback

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from PIL import (
    Image,
    UnidentifiedImageError,
)

from config import (
    MAX_UPLOAD_SIZE,
    MAX_IMAGE_PIXELS,
)

from services.upscale_service import (
    upscale_image,
)

from utils.image import (
    create_input_path,
    create_output_path,
    cleanup,
)

from utils.progress import (
    update_progress,
    get_progress,
    remove_progress,
)


router = APIRouter()


# ============================================================
# ALLOWED IMAGE TYPES
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
# JOB ID VALIDATION
# ============================================================

def validate_job_id(
    job_id: str,
) -> bool:
    """
    Validate frontend-generated job IDs.

    Only simple alphanumeric IDs, underscores and hyphens
    are accepted.
    """

    if not job_id:
        return False

    return bool(
        re.fullmatch(
            r"[A-Za-z0-9_-]{8,100}",
            job_id,
        )
    )


# ============================================================
# FILE EXTENSION VALIDATION
# ============================================================

def validate_extension(
    filename: str,
) -> bool:
    """
    Validate the uploaded filename extension.
    """

    extension = os.path.splitext(
        filename or ""
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image_file(
    path: str,
):
    """
    Validate that the uploaded file is a real supported image.

    Returns:

        width
        height
        format
    """

    try:

        # ------------------------------------------------------
        # Open image
        # ------------------------------------------------------

        with Image.open(path) as image:

            width, height = image.size

            image_format = (
                image.format or ""
            ).upper()

            # --------------------------------------------------
            # Basic dimensions
            # --------------------------------------------------

            if width <= 0 or height <= 0:

                raise ValueError(
                    "Invalid image dimensions."
                )

            # --------------------------------------------------
            # Pixel limit
            # --------------------------------------------------

            pixels = width * height

            if pixels > MAX_IMAGE_PIXELS:

                raise ValueError(
                    "Image resolution is too large. "
                    "Please upload a smaller image."
                )

            # --------------------------------------------------
            # Verify file contents
            # --------------------------------------------------

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
    """
    Return current processing progress.
    """

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

    return get_progress(
        job_id
    )


# ============================================================
# UPSCALE
# ============================================================

@router.post(
    "/upscale"
)
async def upscale(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_id: str = Form(...),
):
    """
    Upload and upscale an image.

    CPU-heavy inference is executed through asyncio.to_thread()
    so that the FastAPI event loop remains responsive for:

        - progress polling
        - health checks
        - other HTTP requests
    """

    input_path = None
    output_path = None

    try:

        # ====================================================
        # 1. VALIDATE JOB ID
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
        # 2. VALIDATE FILENAME
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
        # 3. VALIDATE MIME TYPE
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
        # 4. CREATE TEMPORARY PATHS
        # ====================================================

        input_path = create_input_path(
            filename
        )

        output_path = create_output_path()

        # ====================================================
        # 5. UPLOAD FILE
        # ====================================================

        update_progress(
            job_id,
            5,
            "Uploading image",
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

                # --------------------------------------------
                # File size protection
                # --------------------------------------------

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
                                "Maximum allowed size is "
                                "10 MB."
                            ),
                        },
                    )

                buffer.write(
                    chunk
                )

        # ====================================================
        # 6. CLOSE UPLOAD
        # ====================================================

        await file.close()

        # ====================================================
        # 7. VALIDATE ACTUAL IMAGE
        # ====================================================

        update_progress(
            job_id,
            12,
            "Validating image",
        )

        (
            width,
            height,
            image_format,
        ) = validate_image_file(
            input_path
        )

        # ====================================================
        # 8. PREPARING
        # ====================================================

        update_progress(
            job_id,
            15,
            (
                f"Preparing {width}×{height} "
                f"{image_format} image"
            ),
        )

        # ====================================================
        # 9. CPU-BOUND UPSCALING
        # ====================================================

        """
        IMPORTANT:

        upscale_image() is synchronous and CPU-heavy.

        asyncio.to_thread() prevents ONNX processing from
        blocking the FastAPI event loop.
        """

        await asyncio.to_thread(
            upscale_image,
            input_path,
            output_path,
            job_id,
        )

        # ====================================================
        # 10. SCHEDULE CLEANUP
        # ====================================================

        background_tasks.add_task(
            cleanup,
            input_path,
            output_path,
        )

        background_tasks.add_task(
            remove_progress,
            job_id,
        )

        # ====================================================
        # 11. RETURN OUTPUT
        # ====================================================

        return FileResponse(
            output_path,
            media_type="image/png",
            filename="upscaled.png",
        )

    # ========================================================
    # BUSY SERVER
    # ========================================================

    except RuntimeError as exc:

        message = str(
            exc
        )

        if (
            "currently busy"
            in message.lower()
        ):

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
                        "The upscaler is currently "
                        "processing another image. "
                        "Please try again shortly."
                    ),
                },
            )

        # ----------------------------------------------------
        # Other runtime errors
        # ----------------------------------------------------

        traceback.print_exc()

        cleanup(
            input_path,
            output_path,
        )

        update_progress(
            job_id,
            0,
            "Failed",
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Image processing failed."
                ),
            },
        )

    # ========================================================
    # VALIDATION / USER ERROR
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
                "message": str(
                    exc
                ),
            },
        )

    # ========================================================
    # IMAGE ERRORS
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

    except Exception:

        traceback.print_exc()

        cleanup(
            input_path,
            output_path,
        )

        update_progress(
            job_id,
            0,
            "Failed",
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "Unable to process the image. "
                    "Please try again."
                ),
            },
        )