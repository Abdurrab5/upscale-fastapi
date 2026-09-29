import json
import os
import re
import tempfile
import threading
import time


# ============================================================
# APPLICATION PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# PROGRESS DIRECTORY
# ============================================================

PROGRESS_DIR = os.path.join(
    BASE_DIR,
    "progress",
)


os.makedirs(
    PROGRESS_DIR,
    exist_ok=True,
)

_WRITE_LOCK = threading.Lock()
_LAST_WRITES = {}
_LAST_SWEEP = 0.0
_MIN_PROGRESS_DELTA = 2
_MIN_WRITE_INTERVAL = 0.5
_PROGRESS_TTL_SECONDS = 24 * 60 * 60


# ============================================================
# JOB ID VALIDATION
# ============================================================

def _valid_job_id(
    job_id: str,
) -> bool:
    """
    Validate the job ID supplied by Laravel.

    Laravel generates a UUID such as:

        xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    The allowed pattern also supports letters, numbers,
    underscores and hyphens for compatibility.
    """

    if not job_id:
        return False

    return bool(
        re.fullmatch(
            r"[A-Za-z0-9_-]{8,100}",
            str(job_id),
        )
    )


# ============================================================
# PROGRESS PATH
# ============================================================

def _progress_path(
    job_id: str,
) -> str:
    """
    Return the filesystem path for a job's progress file.
    """

    if not _valid_job_id(
        job_id
    ):
        raise ValueError(
            "Invalid job ID."
        )

    return os.path.join(
        PROGRESS_DIR,
        f"{job_id}.json",
    )


def _write_progress_atomically(target, job_id, payload):
    directory = os.path.dirname(target)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{job_id}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


# ============================================================
# UPDATE PROGRESS
# ============================================================

def update_progress(
    job_id: str,
    percent: int,
    message: str,
    status: str = "processing",
):
    """
    Atomically update progress.

    The frontend will never observe a partially written
    JSON file.

    Status values:

        processing
        completed
        failed
    """

    target = _progress_path(
        job_id
    )

    percent = max(
        0,
        min(
            100,
            int(percent),
        ),
    )

    status = str(
        status or "processing"
    ).strip().lower()

    if status not in {
        "processing",
        "completed",
        "failed",
    }:
        status = "processing"

    now = time.monotonic()
    with _WRITE_LOCK:
        previous = _LAST_WRITES.get(str(job_id))
        important = status in {"completed", "failed"} or percent in {0, 100}
        if not important and previous is not None:
            previous_percent, previous_time = previous
            if (abs(percent - previous_percent) < _MIN_PROGRESS_DELTA
                    and now - previous_time < _MIN_WRITE_INTERVAL):
                return
        # Progress is polled state, not durable history. Opportunistically
        # remove files older than a day so this directory stays bounded.
        global _LAST_SWEEP
        if now - _LAST_SWEEP > 300:
            _LAST_SWEEP = now
            wall_cutoff = time.time() - _PROGRESS_TTL_SECONDS
            try:
                for name in os.listdir(PROGRESS_DIR):
                    if name.endswith((".json", ".tmp")):
                        path = os.path.join(PROGRESS_DIR, name)
                        try:
                            if os.path.getmtime(path) < wall_cutoff:
                                os.remove(path)
                                if name.endswith(".json"):
                                    _LAST_WRITES.pop(name[:-5], None)
                        except OSError:
                            pass
            except OSError:
                pass
        payload = {
            "job_id": str(
                job_id
            ),
            "percent": percent,
            "status": status,
            "message": str(
                message
            ),
        }

        _write_progress_atomically(target, job_id, payload)
        _LAST_WRITES[str(job_id)] = (percent, now)

        if len(_LAST_WRITES) > 4096:
            excess = len(_LAST_WRITES) - 4096
            oldest = sorted(_LAST_WRITES.items(), key=lambda item: item[1][1])[:excess]
            for cached_job, _ in oldest:
                _LAST_WRITES.pop(cached_job, None)


# ============================================================
# GET PROGRESS
# ============================================================

def get_progress(
    job_id: str,
):
    """
    Read the latest progress state.

    If processing has not started yet, return a waiting
    state.

    Progress does NOT contain the output file path or URL.
    The finished image is returned directly by the
    synchronous /upscale endpoint.
    """

    path = _progress_path(
        job_id
    )

    if not os.path.isfile(
        path
    ):

        return {
            "job_id": str(
                job_id
            ),
            "percent": 0,
            "status": "waiting",
            "message": "Waiting",
        }

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        # ----------------------------------------------------
        # Ensure a predictable response structure.
        # ----------------------------------------------------

        return {
            "job_id": str(
                job_id
            ),

            "percent": max(
                0,
                min(
                    100,
                    int(
                        data.get(
                            "percent",
                            0,
                        )
                    ),
                ),
            ),

            "status": str(
                data.get(
                    "status",
                    "processing",
                )
            ),

            "message": str(
                data.get(
                    "message",
                    "Processing",
                )
            ),
        }

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):

        return {
            "job_id": str(
                job_id
            ),
            "percent": 0,
            "status": "processing",
            "message": "Processing",
        }


# ============================================================
# MARK COMPLETED
# ============================================================

def mark_completed(
    job_id: str,
    message: str = "AI enhancement completed",
):
    """
    Mark a job as successfully completed.

    The output image itself is NOT stored here.
    """

    update_progress(
        job_id,
        100,
        message,
        "completed",
    )


# ============================================================
# MARK FAILED
# ============================================================

def mark_failed(
    job_id: str,
    message: str = "AI processing failed",
):
    """
    Mark a job as failed.
    """

    update_progress(
        job_id,
        0,
        message,
        "failed",
    )


# ============================================================
# REMOVE PROGRESS
# ============================================================

def remove_progress(
    job_id: str,
):
    """
    Remove progress state when it is explicitly requested.

    This is intentionally NOT required immediately after a
    successful synchronous response. Keeping the completed
    state briefly allows the frontend's final progress poll
    to see 100% / completed instead of reverting to Waiting.
    """

    try:

        path = _progress_path(
            job_id
        )

        if os.path.isfile(
            path
        ):

            os.remove(
                path
            )

        with _WRITE_LOCK:
            _LAST_WRITES.pop(str(job_id), None)

    except (
        OSError,
        ValueError,
    ):

        pass
