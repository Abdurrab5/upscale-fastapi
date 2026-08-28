from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from services.upscale_service import upscale_image
from utils.image import cleanup
from utils.progress import (
    remove_progress,
    update_progress,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_WORKERS = max(
    1,
    int(
        os.getenv(
            "UPSCALE_MAX_WORKERS",
            "1",
        )
    ),
)

MAX_QUEUE = max(
    1,
    int(
        os.getenv(
            "UPSCALE_MAX_QUEUE",
            "20",
        )
    ),
)


# ============================================================
# EXECUTOR
# ============================================================

_executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS,
    thread_name_prefix="upscale-worker",
)


# ============================================================
# JOB STORAGE
# ============================================================

_jobs: dict[str, dict] = {}

_jobs_lock = threading.Lock()

_submit_lock = threading.Lock()


# ============================================================
# ERRORS
# ============================================================

class QueueFullError(RuntimeError):
    """Raised when the processing queue is full."""


class JobNotFoundError(RuntimeError):
    """Raised when a requested job does not exist."""


class DownloadNotReadyError(RuntimeError):
    """Raised when a job cannot currently be downloaded."""


# ============================================================
# JOB EXECUTION
# ============================================================

def _run_job(
    job_id: str,
    input_path: str,
    output_path: str,
    quality: str,
):
    """
    Execute one queued upscale job.

    IMPORTANT:

    The completed output remains on disk.

    It is NOT deleted here.

    It is deleted only after the download endpoint has
    successfully claimed and served the file.
    """

    try:

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        with _jobs_lock:

            job = _jobs.get(job_id)

            if job is None:
                return

            job["status"] = "processing"

        update_progress(
            job_id,
            20,
            f"Enhancing image to {quality.upper()}",
            status="processing",
        )

        # ----------------------------------------------------
        # AI UPSCALE
        # ----------------------------------------------------

        upscale_image(
            input_path,
            output_path,
            job_id,
            quality,
        )

        # ----------------------------------------------------
        # VERIFY OUTPUT
        # ----------------------------------------------------

        if (
            not os.path.isfile(output_path)
            or os.path.getsize(output_path) <= 0
        ):
            raise RuntimeError(
                "AI processing completed without "
                "creating an output image."
            )

        # ----------------------------------------------------
        # COMPLETED
        # ----------------------------------------------------

        with _jobs_lock:

            job = _jobs.get(job_id)

            if job is not None:

                job["status"] = "completed"

                job["error"] = None

        update_progress(
            job_id,
            100,
            f"{quality.upper()} enhancement completed",
            status="completed",
        )

        print(
            f"[UPSCALE] Completed job: {job_id}",
            flush=True,
        )

    except Exception as exc:

        print(
            f"[UPSCALE] Failed job {job_id}: {exc!r}",
            flush=True,
        )

        # ----------------------------------------------------
        # FAILED
        # ----------------------------------------------------

        with _jobs_lock:

            job = _jobs.get(job_id)

            if job is not None:

                job["status"] = "failed"

                job["error"] = str(exc)

        update_progress(
            job_id,
            100,
            "Image processing failed",
            status="failed",
        )

        # Failed jobs have no downloadable result.
        cleanup(
            input_path,
            output_path,
        )


# ============================================================
# SUBMIT
# ============================================================

def submit_job(
    job_id: str,
    input_path: str,
    output_path: str,
    quality: str,
):
    """
    Add a job to the processing queue.

    The function returns immediately.
    """

    with _submit_lock:

        with _jobs_lock:

            active = sum(
                1
                for job in _jobs.values()
                if job.get("status")
                in {
                    "queued",
                    "processing",
                }
            )

            if active >= MAX_QUEUE:

                raise QueueFullError(
                    "The processing queue is currently full."
                )

            # ------------------------------------------------
            # Prevent duplicate active job IDs.
            # ------------------------------------------------

            existing = _jobs.get(job_id)

            if existing is not None:

                existing_status = existing.get(
                    "status"
                )

                if existing_status in {
                    "queued",
                    "processing",
                    "completed",
                    "downloading",
                }:

                    raise RuntimeError(
                        "This job ID is already in use."
                    )

            _jobs[job_id] = {
                "status": "queued",
                "quality": quality,
                "input_path": input_path,
                "output_path": output_path,
                "error": None,
            }

        update_progress(
            job_id,
            15,
            f"Preparing {quality.upper()} enhancement",
            status="queued",
        )

        try:

            _executor.submit(
                _run_job,
                job_id,
                input_path,
                output_path,
                quality,
            )

        except Exception:

            with _jobs_lock:

                _jobs.pop(
                    job_id,
                    None,
                )

            remove_progress(
                job_id
            )

            raise


# ============================================================
# GET JOB
# ============================================================

def get_job(
    job_id: str,
):
    """
    Return a copy of the current job.
    """

    with _jobs_lock:

        job = _jobs.get(job_id)

        if job is None:
            return None

        return dict(job)


# ============================================================
# CLAIM DOWNLOAD
# ============================================================

def claim_download(
    job_id: str,
):
    """
    Atomically claim a completed job for download.

    Only one request can claim a job.

    Returns the job information needed by the download
    endpoint.
    """

    with _jobs_lock:

        job = _jobs.get(job_id)

        if job is None:

            raise JobNotFoundError(
                "The requested enhancement job no longer exists."
            )

        status = job.get(
            "status"
        )

        if status in {
            "queued",
            "processing",
        }:

            raise DownloadNotReadyError(
                "Your image is still being enhanced."
            )

        if status == "downloading":

            raise DownloadNotReadyError(
                "Your download is already being prepared."
            )

        if status == "failed":

            raise DownloadNotReadyError(
                "This image could not be enhanced."
            )

        if status != "completed":

            raise DownloadNotReadyError(
                "This enhancement is not available for download."
            )

        output_path = job.get(
            "output_path"
        )

        if (
            not output_path
            or not os.path.isfile(output_path)
            or os.path.getsize(output_path) <= 0
        ):

            job["status"] = "failed"

            job["error"] = (
                "The enhanced image is no longer available."
            )

            raise DownloadNotReadyError(
                "The enhanced image is no longer available."
            )

        # ----------------------------------------------------
        # Lock the job against duplicate downloads.
        # ----------------------------------------------------

        job["status"] = "downloading"

        return dict(job)


# ============================================================
# COMPLETE DOWNLOAD
# ============================================================

def complete_download(
    job_id: str,
):
    """
    Permanently consume a completed job.

    Deletes:

        input file
        output file
        progress file
        in-memory job ID
    """

    with _jobs_lock:

        job = _jobs.pop(
            job_id,
            None,
        )

    if job is None:
        return

    cleanup(
        job.get("input_path"),
        job.get("output_path"),
    )

    remove_progress(
        job_id
    )

    print(
        f"[UPSCALE] Consumed job: {job_id}",
        flush=True,
    )


# ============================================================
# ABORT DOWNLOAD
# ============================================================

def abort_download(
    job_id: str,
):
    """
    Release a download claim if serving the file fails.

    The completed output remains available so the user can
    retry the download.
    """

    with _jobs_lock:

        job = _jobs.get(job_id)

        if job is None:
            return

        if job.get("status") == "downloading":

            job["status"] = "completed"