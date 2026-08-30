import threading

import numpy as np

from services.session import get_session
from services.tile_processor import (
    resize_image_array,
    run_ai_pass,
)


# ============================================================
# INFERENCE ENGINE
# ============================================================

class InferenceEngine:
    """
    Thread-safe Real-ESRGAN inference engine.

    The ONNX session is loaded once and reused.

    Only one upscale job is allowed to execute at a time because
    CPU-based Real-ESRGAN inference can consume substantial
    memory and CPU resources.
    """

    def __init__(self):

        self.session = get_session()

        self.job_lock = threading.Lock()

    # ========================================================
    # UPSCALE
    # ========================================================

    def upscale(
        self,
        tensor,
        target_width,
        target_height,
        ai_passes=1,
        progress_callback=None,
    ):
        """
        Run the requested number of Real-ESRGAN passes.

        ai_passes=0:
            Directly resize the source image.

        ai_passes>=1:
            Run Real-ESRGAN x4 inference and produce the exact
            requested final dimensions.
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

            # ====================================================
            # VALIDATE
            # ====================================================

            if tensor is None:

                raise ValueError(
                    "AI input tensor is required."
                )

            if not isinstance(
                tensor,
                np.ndarray,
            ):

                raise TypeError(
                    "AI input tensor must be a NumPy array."
                )

            if tensor.ndim != 4:

                raise ValueError(
                    "AI input tensor must have NCHW shape."
                )

            target_width = int(
                target_width
            )

            target_height = int(
                target_height
            )

            ai_passes = int(
                ai_passes
            )

            if target_width <= 0:

                raise ValueError(
                    "Invalid target width."
                )

            if target_height <= 0:

                raise ValueError(
                    "Invalid target height."
                )

            if ai_passes < 0:

                raise ValueError(
                    "ai_passes cannot be negative."
                )

            # ====================================================
            # NO AI REQUIRED
            # ====================================================
            #
            # Used when the source is already larger than the
            # requested target.
            #
            # Example:
            #
            # 3707x3001 -> 2048x1658
            #
            # No Real-ESRGAN inference is necessary.
            #

            if ai_passes == 0:

                if progress_callback:

                    progress_callback(0)

                source = (
                    tensor[0]
                    .transpose(
                        1,
                        2,
                        0,
                    )
                )

                source = np.clip(
                    source,
                    0.0,
                    1.0,
                )

                source = (
                    source
                    * 255.0
                ).astype(
                    np.uint8
                )

                result = resize_image_array(
                    source,
                    target_width,
                    target_height,
                )

                del source

                if progress_callback:

                    progress_callback(100)

                return result

            # ====================================================
            # AI PASSES
            # ====================================================

            current = tensor

            for pass_number in range(
                ai_passes
            ):

                _, _, height, width = (
                    current.shape
                )

                remaining_passes = (
                    ai_passes
                    - pass_number
                    - 1
                )

                # ------------------------------------------------
                # Intermediate AI pass
                # ------------------------------------------------

                if remaining_passes > 0:

                    pass_width = min(
                        target_width,
                        width * 4,
                    )

                    pass_height = min(
                        target_height,
                        height * 4,
                    )

                # ------------------------------------------------
                # Final AI pass
                # ------------------------------------------------

                else:

                    pass_width = target_width

                    pass_height = target_height

                def pass_progress(
                    percent,
                    current_pass=pass_number,
                ):

                    if not progress_callback:

                        return

                    completed_before = (
                        current_pass
                        * 100
                    )

                    total_progress = (
                        completed_before
                        + percent
                    ) / ai_passes

                    progress_callback(
                        min(
                            100,
                            int(
                                total_progress
                            ),
                        )
                    )

                result = run_ai_pass(
                    self.session,
                    current,
                    pass_width,
                    pass_height,
                    progress_callback=pass_progress,
                )

                # ------------------------------------------------
                # Intermediate result → next AI pass
                # ------------------------------------------------

                if (
                    pass_number
                    < ai_passes - 1
                ):

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

                # ------------------------------------------------
                # Final result
                # ------------------------------------------------

                else:

                    return result

            # This should never be reached when ai_passes >= 1.

            raise RuntimeError(
                "AI upscaling produced no result."
            )

        finally:

            self.job_lock.release()


# ============================================================
# SINGLETON ENGINE
# ============================================================

_engine = None

_engine_lock = threading.Lock()


def get_engine():
    """
    Return the shared InferenceEngine instance.

    The ONNX session and engine are initialized only once.
    """

    global _engine

    if _engine is not None:

        return _engine

    with _engine_lock:

        if _engine is None:

            _engine = InferenceEngine()

        return _engine