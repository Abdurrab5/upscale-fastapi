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
    "hd": 2048,
    "4k": 4096,
}


DEFAULT_QUALITY = "4k"


ALLOWED_QUALITY_VALUES = {
    "hd",
    "4k",
}


# ============================================================
# QUALITY ALIASES
# ============================================================

QUALITY_ALIASES = {
    # HD
    "hd": "hd",
    "2k": "hd",
    "2048": "hd",
    "2k-hd": "hd",

    # 4K
    "4k": "4k",
    "4096": "4k",
    "4k-hd": "4k",
}


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
# OUTPUT FORMATS
# ============================================================

ALLOWED_OUTPUT_FORMATS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
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
    quality: str | None,
) -> str:
    """
    Normalize Laravel/frontend quality values.

    Supported final modes:

        hd  -> 2048px longest edge
        4k  -> 4096px longest edge

    Legacy 2K/2048 values are accepted as HD
    for compatibility.
    """

    value = str(
        quality or DEFAULT_QUALITY
    ).strip().lower()

    value = QUALITY_ALIASES.get(
        value,
        value,
    )

    if value not in ALLOWED_QUALITY_VALUES:
        return DEFAULT_QUALITY

    return value


# ============================================================
# OUTPUT FORMAT NORMALIZATION
# ============================================================

def normalize_output_format(
    output_format: str | None,
    source_format: str | None = None,
) -> str:
    """
    Normalize output format.

    Explicit Laravel output_format wins.

    If omitted, preserve the original supported
    image format where possible.

    PNG therefore remains PNG by default,
    preserving transparency.
    """

    value = str(
        output_format or ""
    ).strip().lower()

    if value == "jpg":
        value = "jpeg"

    if value in {
        "png",
        "jpeg",
        "webp",
    }:
        return value

    source = str(
        source_format or ""
    ).strip().lower()

    if source == "jpg":
        source = "jpeg"

    if source in {
        "png",
        "jpeg",
        "webp",
    }:
        return source

    return "png"


# ============================================================
# TARGET RESOLUTION VALIDATION
# ============================================================

def is_target_supported(
    width: int,
    height: int,
) -> bool:

    if width <= 0:
        return False

    if height <= 0:
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
# RESOLVE TARGET
# ============================================================

def resolve_target(
    source_width: int,
    source_height: int,
    quality: str,
):
    """
    Resolve final output dimensions from quality.

    HD:
        2048px longest edge

    4K:
        4096px longest edge

    Aspect ratio is always preserved.
    """

    quality = normalize_quality(
        quality
    )

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

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not is_target_supported(
        target_width,
        target_height,
    ):

        # Safe fallback to HD.

        quality = "hd"

        target_longest = QUALITY_MODES[
            quality
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

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if not is_target_supported(
        target_width,
        target_height,
    ):

        raise ValueError(
            "Requested output resolution "
            "cannot be safely generated."
        )

    return (
        quality,
        target_width,
        target_height,
    )


# ============================================================
# RESOLVE TARGET FROM EDGE
# ============================================================

def resolve_target_from_edge(
    source_width: int,
    source_height: int,
    target_longest_edge: int,
):
    """
    Laravel compatibility.

    Supported:

        2048 -> HD
        4096 -> 4K
    """

    edge_to_quality = {
        2048: "hd",
        4096: "4k",
    }

    if target_longest_edge not in edge_to_quality:

        raise ValueError(
            "target_longest_edge must be "
            "2048 or 4096."
        )

    return resolve_target(
        source_width,
        source_height,
        edge_to_quality[
            target_longest_edge
        ],
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image_file(
    path: str,
):
    """
    Validate actual image contents.

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
            # Pixel safety
            # ------------------------------------------------

            pixels = (
                width * height
            )

            if pixels > MAX_IMAGE_PIXELS:

                raise ValueError(
                    "Image resolution is too large."
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
# SYNCHRONOUS UPSCALE
# ============================================================

@router.post(
    "/upscale"
)
async def upscale(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    quality: str = Form(DEFAULT_QUALITY),
    target_longest_edge: int | None = Form(None),
    output_format: str | None = Form(None),
):

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
        # 5. CREATE TEMPORARY PATHS
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
                                "Image file is too large."
                            ),
                        },
                    )

                buffer.write(
                    chunk
                )

        await file.close()


        # ====================================================
        # 7. VALIDATE IMAGE
        # ====================================================

        update_progress(
            job_id,
            12,
            "Validating image",
        )

        (
            source_width,
            source_height,
            source_format,
        ) = validate_image_file(
            input_path
        )


        # ====================================================
        # 8. RESOLVE TARGET FROM QUALITY
        # ====================================================

        (
            final_quality,
            target_width,
            target_height,
        ) = resolve_target(
            source_width,
            source_height,
            requested_quality,
        )


        # ====================================================
        # 9. TARGET LONGEST EDGE OVERRIDE
        # ====================================================

        if target_longest_edge is not None:

            (
                final_quality,
                target_width,
                target_height,
            ) = resolve_target_from_edge(
                source_width,
                source_height,
                target_longest_edge,
            )


        # ====================================================
        # 10. OUTPUT FORMAT
        # ====================================================

        final_output_format = (
            normalize_output_format(
                output_format,
                source_format,
            )
        )


        # ====================================================
        # 11. PREPARING
        # ====================================================

        update_progress(
            job_id,
            15,
            (
                f"Preparing "
                f"{final_quality.upper()} "
                f"{target_width}×{target_height}"
            ),
        )


        # ====================================================
        # 12. AI UPSCALING
        # ====================================================

        update_progress(
            job_id,
            20,
            (
                f"Running Real-ESRGAN "
                f"{final_quality.upper()} AI enhancement"
            ),
        )

        await asyncio.to_thread(
            upscale_image,
            input_path,
            output_path,
            job_id,
            final_quality,
            target_width,
            target_height,
            final_output_format,
        )


        # ====================================================
        # 13. VERIFY OUTPUT FILE
        # ====================================================

        if (
            not output_path
            or not os.path.isfile(
                output_path
            )
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
                "Generated output image is empty."
            )


        # ====================================================
        # 14. VERIFY GENERATED IMAGE
        # ====================================================

        try:

            with Image.open(
                output_path
            ) as output_image:

                actual_width, actual_height = (
                    output_image.size
                )

                if (
                    actual_width != target_width
                    or actual_height != target_height
                ):

                    raise RuntimeError(
                        "Generated image dimensions do not "
                        "match the requested output resolution."
                    )

        except UnidentifiedImageError:

            raise RuntimeError(
                "Generated output is not a valid image."
            )


        # ====================================================
        # 15. COMPLETE
        # ====================================================

        update_progress(
            job_id,
            100,
            (
                f"{final_quality.upper()} "
                "AI enhancement completed"
            ),
        )


        # ====================================================
        # 16. MEDIA TYPE
        # ====================================================

        media_types = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }

        media_type = media_types[
            final_output_format
        ]


        # ====================================================
        # 17. FILE EXTENSION
        # ====================================================

        extension = (
            "jpg"
            if final_output_format == "jpeg"
            else final_output_format
        )


        # ====================================================
        # 18. RETURN ACTUAL OUTPUT FILE
        # ====================================================

        return FileResponse(
            output_path,
            media_type=media_type,
            filename=(
                f"upscaled-"
                f"{final_quality}."
                f"{extension}"
            ),
            background=None,
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
                "message": str(exc),
            },
        )


    # ========================================================
    # PROCESSING ERROR
    # ========================================================

    except RuntimeError as exc:

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