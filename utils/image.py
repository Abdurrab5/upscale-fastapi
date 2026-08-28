import logging
import os
import uuid


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


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
# INPUT PATH
# ============================================================

def create_input_path(
    filename: str,
) -> str:
    """
    Create a unique temporary input path.

    The original filename is used only to preserve the
    extension.

    The actual filesystem name is generated using UUID,
    preventing:

        - path traversal
        - filename collisions
        - user-to-user collisions
        - concurrent request conflicts
    """

    extension = os.path.splitext(
        filename or ""
    )[1].lower()

    # --------------------------------------------------------
    # Only keep a safe extension.
    # --------------------------------------------------------

    if extension not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:

        extension = ".tmp"

    return os.path.join(
        TEMP_DIR,
        f"{uuid.uuid4().hex}{extension}",
    )


# ============================================================
# OUTPUT PATH
# ============================================================

def create_output_path() -> str:
    """
    Create a unique temporary output path.

    Every request receives its own output file.
    """

    return os.path.join(
        TEMP_DIR,
        f"{uuid.uuid4().hex}.png",
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

    Cleanup failures are logged but never allowed to crash
    the request.
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
                "Unable to remove temporary file %s: %s",
                path,
                exc,
            )


# ============================================================
# JOB TEMP DIRECTORY CLEANUP
# ============================================================

def cleanup_temp_directory():
    """
    Remove leftover temporary files.

    This is useful for cleaning files left behind after:

        - process crashes
        - container restarts
        - request timeouts
        - unexpected exceptions
    """

    try:

        for filename in os.listdir(
            TEMP_DIR
        ):

            path = os.path.join(
                TEMP_DIR,
                filename,
            )

            if os.path.isfile(
                path
            ):

                try:

                    os.remove(
                        path
                    )

                except OSError as exc:

                    logger.warning(
                        "Unable to remove temporary file %s: %s",
                        path,
                        exc,
                    )

    except OSError as exc:

        logger.warning(
            "Unable to inspect temporary directory: %s",
            exc,
        )
 
