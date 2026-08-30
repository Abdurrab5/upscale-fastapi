import logging
import os
import uuid


# ============================================================
# APPLICATION PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# TEMPORARY DIRECTORY
# ============================================================

TEMP_DIR = os.path.join(
    BASE_DIR,
    "temp",
)


os.makedirs(
    TEMP_DIR,
    exist_ok=True,
)


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    __name__
)


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

SUPPORTED_INPUT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


SUPPORTED_OUTPUT_FORMATS = {
    "png",
    "jpeg",
    "webp",
}


# ============================================================
# INPUT PATH
# ============================================================

def create_input_path(
    filename: str,
) -> str:
    """
    Create a unique temporary input path.

    The original filename is NEVER used directly as the
    filesystem name.

    This prevents:

        - path traversal
        - filename collisions
        - unsafe filenames
    """

    extension = os.path.splitext(
        filename or ""
    )[1].lower()

    # --------------------------------------------------------
    # Retain only supported image extensions.
    # --------------------------------------------------------

    if extension not in SUPPORTED_INPUT_EXTENSIONS:
        extension = ".tmp"

    # --------------------------------------------------------
    # UUID-based filename.
    # --------------------------------------------------------

    return os.path.join(
        TEMP_DIR,
        f"{uuid.uuid4().hex}{extension}",
    )


# ============================================================
# OUTPUT FORMAT NORMALIZATION
# ============================================================

def normalize_output_format(
    output_format: str | None,
) -> str:
    """
    Normalize the requested output format.

    Supported:

        png
        jpg
        jpeg
        webp

    Internally:

        jpg -> jpeg

    Invalid or missing values fall back to PNG.
    """

    value = str(
        output_format or "png"
    ).strip().lower()

    if value == "jpg":
        value = "jpeg"

    if value not in SUPPORTED_OUTPUT_FORMATS:
        value = "png"

    return value


# ============================================================
# OUTPUT PATH
# ============================================================

def create_output_path(
    output_format: str = "png",
) -> str:
    """
    Create a unique temporary output path.

    Supported formats:

        png
        jpeg
        jpg
        webp

    JPG is normalized internally to JPEG and receives
    a .jpg filesystem extension.
    """

    output_format = normalize_output_format(
        output_format
    )

    # --------------------------------------------------------
    # Filesystem extension.
    # --------------------------------------------------------

    extension = (
        "jpg"
        if output_format == "jpeg"
        else output_format
    )

    # --------------------------------------------------------
    # UUID-based filename.
    # --------------------------------------------------------

    return os.path.join(
        TEMP_DIR,
        f"{uuid.uuid4().hex}.{extension}",
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup(
    *files,
):
    """
    Safely remove temporary files.

    Missing files are ignored.

    Cleanup failures are logged but do not crash the
    request.
    """

    for path in files:

        if not path:
            continue

        try:

            if os.path.isfile(
                path
            ):

                os.remove(
                    path
                )

        except OSError as exc:

            logger.warning(
                "Unable to remove temporary "
                "file %s: %s",
                path,
                exc,
            )