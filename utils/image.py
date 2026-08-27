import logging
import os
import uuid


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


logger = logging.getLogger(__name__)


def create_input_path(
    filename: str,
) -> str:
    """
    Create a unique temporary input path.

    The original filename is never used as the actual filesystem
    name, preventing path traversal and filename collisions.
    """

    extension = os.path.splitext(
        filename or ""
    )[1].lower()

    return os.path.join(
        TEMP_DIR,
        f"{uuid.uuid4().hex}{extension}",
    )


def create_output_path() -> str:
    """
    Create a unique temporary output path.
    """

    return os.path.join(
        TEMP_DIR,
        f"{uuid.uuid4().hex}.png",
    )


def cleanup(*files):
    """
    Safely remove temporary files.
    """

    for path in files:

        if not path:
            continue

        try:

            if os.path.isfile(path):
                os.remove(path)

        except OSError as exc:

            logger.warning(
                "Unable to remove temporary file %s: %s",
                path,
                exc,
            )