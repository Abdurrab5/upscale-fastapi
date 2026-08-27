import threading

from services.session import get_session
from services.tile_processor import (
    run_ai_pass,
)


class InferenceEngine:

    def __init__(self):

        self.session = get_session()

        self.job_lock = threading.Lock()


    def upscale(
        self,
        tensor,
        target_width,
        target_height,
        ai_passes=1,
        progress_callback=None,
    ):

        acquired = self.job_lock.acquire(
            blocking=False
        )

        if not acquired:

            raise RuntimeError(
                "The upscaling server is currently busy. "
                "Please try again in a moment."
            )

        try:

            current = tensor

            for pass_number in range(
                ai_passes
            ):

                # ------------------------------------------------
                # Determine current dimensions
                # ------------------------------------------------

                _, _, height, width = (
                    current.shape
                )

                remaining_passes = (
                    ai_passes
                    - pass_number
                    - 1
                )

                if remaining_passes > 0:

                    # Let this pass produce a useful
                    # intermediate enlargement.

                    pass_width = min(
                        target_width,
                        width * 4,
                    )

                    pass_height = min(
                        target_height,
                        height * 4,
                    )

                else:

                    pass_width = target_width
                    pass_height = target_height

                def pass_progress(
                    percent,
                ):

                    if progress_callback:

                        completed_before = (
                            pass_number
                            * 100
                        )

                        total_progress = (
                            completed_before
                            + percent
                        ) / ai_passes

                        progress_callback(
                            int(
                                total_progress
                            )
                        )

                result = run_ai_pass(
                    self.session,
                    current,
                    pass_width,
                    pass_height,
                    progress_callback=pass_progress,
                )

                if (
                    pass_number
                    < ai_passes - 1
                ):

                    # HWC uint8 → NCHW float32

                    import numpy as np

                    current = (
                        result
                        .transpose(
                            2,
                            0,
                            1,
                        )
                        .astype(
                            np.float32
                        )
                        / 255.0
                    )

                    current = np.expand_dims(
                        current,
                        axis=0,
                    )

                    del result

                else:

                    return result

        finally:

            self.job_lock.release()


_engine = None
_engine_lock = threading.Lock()


def get_engine():

    global _engine

    if _engine is not None:
        return _engine

    with _engine_lock:

        if _engine is None:
            _engine = InferenceEngine()

        return _engine