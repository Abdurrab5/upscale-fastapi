import asyncio
import os
import re
import traceback

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
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
    MAX_OUTPUT_DIMENSION,
    MAX_OUTPUT_PIXELS,
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
# QUALITY MODES
# ============================================================

QUALITY_MODES = {
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
}

DEFAULT_QUALITY = "4k"

ALLOWED_QUALITY_VALUES = set(
    QUALITY_MODES.keys()
)


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

    extension = os.path.splitext(
        filename or ""
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# QUALITY NORMALIZATION
# ============================================================

def normalize_quality(
    quality: str,
) -> str:
    """
    Normalize the frontend quality selection.

    Supported values:

        2k
        4k
        8k

    There is intentionally NO auto mode.

    Invalid or unavailable selections fall back to 4K.
    """

    value = str(
        quality or DEFAULT_QUALITY
    ).strip().lower()

    # --------------------------------------------------------
    # Normalize common frontend variations
    # --------------------------------------------------------

    aliases = {
        "2K": "2k",
        "2048": "2k",
        "2k-hd": "2k",
        "hd": "2k",

        "4K": "4k",
        "4096": "4k",
        "4k-hd": "4k",

        "8K": "8k",
        "8192": "8k",
        "8k-hd": "8k",
    }

    value = aliases.get(
        value,
        value,
    )

    # --------------------------------------------------------
    # Only 2K / 4K / 8K
    # --------------------------------------------------------

    if value not in ALLOWED_QUALITY_VALUES:
        return DEFAULT_QUALITY

    return value


# ============================================================
# TARGET RESOLUTION VALIDATION
# ============================================================

def is_target_supported(
    width: int,
    height: int,
) -> bool:
    """
    Verify that the requested output dimensions are within
    server safety limits.
    """

    if width <= 0 or height <= 0:
        return False

    if width > MAX_OUTPUT_DIMENSION:
        return False

    if height > MAX_OUTPUT_DIMENSION:
        return False

    pixels = width * height

    if pixels > MAX_OUTPUT_PIXELS:
        return False

    return True


# ============================================================
# RESOLVE REQUESTED QUALITY
# ============================================================

def resolve_requested_quality(
    source_width: int,
    source_height: int,
    quality: str,
) -> str:
    """
    Select the requested output quality.

    The longest edge is used as the target.

    Example:

        Source:
            826 × 1062

        2K:
            ~1595 × 2048

        4K:
            ~3189 × 4096

        8K:
            ~6378 × 8192

    Aspect ratio is preserved.

    If the selected quality cannot be safely produced
    according to server limits, 4K is used as fallback.
    """

    requested = normalize_quality(
        quality
    )

    # --------------------------------------------------------
    # Try requested quality first
    # --------------------------------------------------------

    target_longest = QUALITY_MODES[
        requested
    ]

    source_longest = max(
        source_width,
        source_height,
    )

    # Preserve aspect ratio.
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

    # --------------------------------------------------------
    # Requested target is supported
    # --------------------------------------------------------

    if is_target_supported(
        target_width,
        target_height,
    ):
        return requested

    # --------------------------------------------------------
    # Requested target unavailable
    #
    # Fallback to 4K.
    # --------------------------------------------------------

    return "4k"


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

        with Image.open(path) as image:

            width, height = image.size

            image_format = (
                image.format or ""
            ).upper()

            # ------------------------------------------------
            # Dimensions
            # ------------------------------------------------

            if width <= 0 or height <= 0:

                raise ValueError(
                    "Invalid image dimensions."
                )

            # ------------------------------------------------
            # Pixel limit
            # ------------------------------------------------

            pixels = (
                width
                * height
            )

            if pixels > MAX_IMAGE_PIXELS:

                raise ValueError(
                    "Image resolution is too large. "
                    "Please upload a smaller image."
                )

            # ------------------------------------------------
            # Verify actual file
            # ------------------------------------------------

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
    file: UploadFile = File(...),
    job_id: str = Form(...),
    quality: str = Form(DEFAULT_QUALITY),
):
    """
    AI image enhancement/upscaling endpoint.

    Supported quality modes:

        2K
        4K
        8K

    There is intentionally NO auto mode.

    The requested quality represents the FINAL OUTPUT
    resolution target.

    It does NOT mean:

        source × 2
        source × 4
        source × 8

    RealESRGAN internally performs its x4 AI inference,
    after which the result is mapped into the requested
    final resolution.
    """

    input_path = None
    output_path = None

    try:

        # ====================================================
        # 1. JOB ID
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
        # 2. QUALITY
        # ====================================================

        requested_quality = normalize_quality(
            quality
        )

        # ====================================================
        # 3. FILE NAME
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
        # 4. MIME TYPE
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
        # 5. TEMPORARY PATHS
        # ====================================================

        input_path = create_input_path(
            filename
        )

        output_path = create_output_path()

        # ====================================================
        # 6. UPLOAD
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
        # 7. CLOSE UPLOAD
        # ====================================================

        await file.close()

        # ====================================================
        # 8. VALIDATE IMAGE
        # ====================================================

        update_progress(
            job_id,
            12,
            "Validating image",
        )

        (
            source_width,
            source_height,
            image_format,
        ) = validate_image_file(
            input_path
        )

        # ====================================================
        # 9. RESOLVE QUALITY
        # ====================================================

        final_quality = resolve_requested_quality(
            source_width,
            source_height,
            requested_quality,
        )

        # ====================================================
        # 10. TARGET LONGEST EDGE
        # ====================================================

        target_longest = QUALITY_MODES[
            final_quality
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
                    source_width
                    * scale
                )
            ),
        )

        target_height = max(
            2,
            int(
                round(
                    source_height
                    * scale
                )
            ),
        )

        # ====================================================
        # 11. FINAL SAFETY CHECK
        # ====================================================

        if not is_target_supported(
            target_width,
            target_height,
        ):

            # ------------------------------------------------
            # Last-resort 4K target.
            # ------------------------------------------------

            final_quality = "4k"

            target_longest = QUALITY_MODES[
                final_quality
            ]

            scale = (
                target_longest
                / source_longest
            )

            target_width = max(
                2,
                int(
                    round(
                        source_width
                        * scale
                    )
                ),
            )

            target_height = max(
                2,
                int(
                    round(
                        source_height
                        * scale
                    )
                ),
            )

            if not is_target_supported(
                target_width,
                target_height,
            ):

                raise ValueError(
                    "The requested image resolution "
                    "cannot be safely generated by "
                    "the current server."
                )

        # ====================================================
        # 12. PREPARING
        # ====================================================

        update_progress(
            job_id,
            15,
            (
                f"Preparing {final_quality.upper()} "
                f"AI output "
                f"{target_width}×{target_height}"
            ),
        )

        # ====================================================
        # 13. AI UPSCALING
        # ====================================================

        update_progress(
            job_id,
            20,
            (
                f"Starting {final_quality.upper()} "
                "AI enhancement"
            ),
        )

        await asyncio.to_thread(
            upscale_image,
            input_path,
            output_path,
            job_id,
            final_quality,
        )

        # ====================================================
        # 14. VERIFY OUTPUT
        # ====================================================

        if not output_path or not os.path.isfile(
            output_path
        ):

            raise RuntimeError(
                "AI processing completed without "
                "creating an output image."
            )

        output_size = os.path.getsize(
            output_path
        )

        if output_size <= 0:

            raise RuntimeError(
                "The generated output image is empty."
            )

        # ====================================================
        # 15. RETURN OUTPUT
        # ====================================================

        update_progress(
            job_id,
            100,
            (
                f"{final_quality.upper()} AI enhancement "
                "completed"
            ),
        )

        return FileResponse(
            output_path,
            media_type="image/png",
            filename=(
                f"upscaled-{final_quality}.png"
            ),
            background=None,
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
                        "The AI upscaler is currently "
                        "processing another image. "
                        "Please try again shortly."
                    ),
                },
            )

        traceback.print_exc()

        cleanup(
            input_path,
            output_path,
        )

        update_progress(
            job_id,
            0,
            "AI processing failed",
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "AI image processing failed. "
                    "Please try again."
                ),
            },
        )

    # ========================================================
    # VALIDATION ERROR
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

    except Exception:

        traceback.print_exc()

        cleanup(
            input_path,
            output_path,
        )

        update_progress(
            job_id,
            0,
            "Processing failed",
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