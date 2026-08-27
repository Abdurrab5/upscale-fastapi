import json
import os
import tempfile


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROGRESS_DIR = os.path.join(
    BASE_DIR,
    "progress",
)

os.makedirs(
    PROGRESS_DIR,
    exist_ok=True,
)


def _progress_path(
    job_id: str,
) -> str:

    return os.path.join(
        PROGRESS_DIR,
        f"{job_id}.json",
    )


def update_progress(
    job_id: str,
    percent: int,
    message: str,
):
    """
    Atomically update progress.

    The frontend will never observe a partially written JSON file.
    """

    percent = max(
        0,
        min(
            100,
            int(percent),
        ),
    )

    target = _progress_path(
        job_id
    )

    directory = os.path.dirname(
        target
    )

    fd, temporary = tempfile.mkstemp(
        prefix=f".{job_id}.",
        suffix=".tmp",
        dir=directory,
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "percent": percent,
                    "message": str(message),
                },
                file,
                separators=(
                    ",",
                    ":",
                ),
            )

            file.flush()
            os.fsync(
                file.fileno()
            )

        os.replace(
            temporary,
            target,
        )

    except Exception:

        try:
            os.remove(
                temporary
            )
        except OSError:
            pass

        raise


def get_progress(
    job_id: str,
):
    """
    Read the latest progress state.
    """

    path = _progress_path(
        job_id
    )

    if not os.path.isfile(path):

        return {
            "percent": 0,
            "message": "Waiting",
        }

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return {
            "percent": 0,
            "message": "Processing",
        }


def remove_progress(
    job_id: str,
):
    """
    Remove progress state after completion.
    """

    path = _progress_path(
        job_id
    )

    try:

        if os.path.isfile(path):
            os.remove(path)

    except OSError:
        pass