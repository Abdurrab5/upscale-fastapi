import threading

from services.session import get_session
from services.tile_processor import merge_tiles


class InferenceEngine:

    def __init__(self):
        self.session = get_session()

        # Only one complete upscale operation at a time.
        #
        # This protects a 1 GB shared hosting environment from
        # simultaneous CPU/RAM-heavy ONNX jobs.
        self.job_lock = threading.Lock()

    def upscale(
        self,
        tensor,
        tile_callback,
        progress_callback=None,
    ):
        """
        Run one complete tiled upscale job.

        Only one job is allowed to use the inference engine at a time.
        """

        acquired = self.job_lock.acquire(
            blocking=False
        )

        if not acquired:
            raise RuntimeError(
                "The upscaling server is currently busy. "
                "Please try again in a moment."
            )

        try:

            return merge_tiles(
                self.session,
                tensor,
                tile_callback=tile_callback,
                progress_callback=progress_callback,
            )

        finally:

            self.job_lock.release()


_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """
    Thread-safe singleton inference engine.
    """

    global _engine

    if _engine is not None:
        return _engine

    with _engine_lock:

        if _engine is None:
            _engine = InferenceEngine()

        return _engine