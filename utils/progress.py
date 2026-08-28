import json
import os
import tempfile
import time

from config import (
    PROGRESS_DIR,
    PROGRESS_TTL_SECONDS,
)


# ============================================================
# PATH
# ============================================================

def _progress_path(
    job_id: str,
) -> str:

    safe_job_id = (
        str(job_id)
        .strip()
    )

    return os.path.join(
        PROGRESS_DIR,
        f"{safe_job_id}.json",
    )


# ============================================================
# UPDATE
# ============================================================

def update_progress(
    job_id: str,
    percent: int,
    message: str,
    status: str = "processing",
):

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

        payload = {
            "percent": percent,
            "status": status,
            "message": str(message),
            "updated_at": time.time(),
        }

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
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


# ============================================================
# GET
# ============================================================

def get_progress(
    job_id: str,
):

    path = _progress_path(
        job_id
    )

    if not os.path.isfile(path):

        return {
            "percent": 0,
            "status": "queued",
            "message": "Waiting",
        }

    try:

        if (
            time.time()
            - os.path.getmtime(path)
            > PROGRESS_TTL_SECONDS
        ):

            remove_progress(
                job_id
            )

            return {
                "percent": 0,
                "status": "expired",
                "message": "Job expired",
            }

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return {
            "percent": 0,
            "status": "processing",
            "message": "Processing",
        }


# ============================================================
# REMOVE
# ============================================================

def remove_progress(
    job_id: str,
):

    path = _progress_path(
        job_id
    )

    try:

        if os.path.isfile(path):
            os.remove(path)

    except OSError:
        pass