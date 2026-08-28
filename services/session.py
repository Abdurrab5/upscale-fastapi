import os
import threading

import onnxruntime as ort

from config import (
    CPU_THREADS,
    FAST_X2_MODEL_PATH,
    INTER_OP_THREADS,
    REAL_ESRGAN_MODEL_PATH,
)


_fast_session = None
_esrgan_session = None

_fast_lock = threading.Lock()
_esrgan_lock = threading.Lock()


# ============================================================
# SESSION OPTIONS
# ============================================================

def _create_options():

    options = ort.SessionOptions()

    options.intra_op_num_threads = max(
        1,
        CPU_THREADS,
    )

    options.inter_op_num_threads = max(
        1,
        INTER_OP_THREADS,
    )

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    options.enable_cpu_mem_arena = True

    options.enable_mem_pattern = True

    options.log_severity_level = 3

    return options


# ============================================================
# FAST X2
# ============================================================

def get_fast_session():

    global _fast_session

    if _fast_session is not None:
        return _fast_session

    with _fast_lock:

        if _fast_session is not None:
            return _fast_session

        if not os.path.isfile(
            FAST_X2_MODEL_PATH
        ):

            raise FileNotFoundError(
                "Fast x2 model not found: "
                f"{FAST_X2_MODEL_PATH}"
            )

        _fast_session = (
            ort.InferenceSession(
                FAST_X2_MODEL_PATH,
                sess_options=_create_options(),
                providers=[
                    "CPUExecutionProvider",
                ],
            )
        )

        return _fast_session


# ============================================================
# REAL-ESRGAN X4
# ============================================================

def get_esrgan_session():

    global _esrgan_session

    if _esrgan_session is not None:
        return _esrgan_session

    with _esrgan_lock:

        if _esrgan_session is not None:
            return _esrgan_session

        if not os.path.isfile(
            REAL_ESRGAN_MODEL_PATH
        ):

            raise FileNotFoundError(
                "Real-ESRGAN model not found: "
                f"{REAL_ESRGAN_MODEL_PATH}"
            )

        _esrgan_session = (
            ort.InferenceSession(
                REAL_ESRGAN_MODEL_PATH,
                sess_options=_create_options(),
                providers=[
                    "CPUExecutionProvider",
                ],
            )
        )

        return _esrgan_session