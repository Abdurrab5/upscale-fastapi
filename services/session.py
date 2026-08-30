
import os
import threading

import onnxruntime as ort

from config import (
    CPU_THREADS,
    MODEL_PATH,
)


# ============================================================
# GLOBAL SESSION
# ============================================================

_session = None

_session_lock = threading.Lock()


# ============================================================
# GET ONNX SESSION
# ============================================================

def get_session():
    """
    Return the shared ONNX Runtime inference session.

    The session is created lazily on the first request and
    reused for subsequent requests.

    Configuration is optimized for the CPU fallback service:

        - CPUExecutionProvider only
        - Sequential execution
        - Limited intra-op threads
        - Single inter-op thread
        - Full graph optimization
        - CPU memory arena enabled
        - Memory pattern enabled

    A lock prevents multiple concurrent requests from creating
    duplicate ONNX Runtime sessions.
    """

    global _session

    # ========================================================
    # FAST PATH
    # ========================================================

    if _session is not None:
        return _session

    # ========================================================
    # THREAD-SAFE INITIALIZATION
    # ========================================================

    with _session_lock:

        # Another request may have initialized the session
        # while this request was waiting for the lock.

        if _session is not None:
            return _session

        # ====================================================
        # MODEL VALIDATION
        # ====================================================

        model_path = os.path.abspath(
            MODEL_PATH
        )

        if not os.path.isfile(
            model_path
        ):

            raise FileNotFoundError(
                f"ONNX model not found: {model_path}"
            )

        if not os.access(
            model_path,
            os.R_OK,
        ):

            raise PermissionError(
                f"ONNX model is not readable: {model_path}"
            )

        # ====================================================
        # SESSION OPTIONS
        # ====================================================

        options = ort.SessionOptions()

        # ----------------------------------------------------
        # CPU THREADING
        # ----------------------------------------------------

        options.intra_op_num_threads = max(
            1,
            int(CPU_THREADS),
        )

        options.inter_op_num_threads = 1

        # ----------------------------------------------------
        # EXECUTION MODE
        # ----------------------------------------------------

        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        # ----------------------------------------------------
        # GRAPH OPTIMIZATION
        # ----------------------------------------------------

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # ----------------------------------------------------
        # MEMORY OPTIMIZATION
        # ----------------------------------------------------

        options.enable_cpu_mem_arena = True

        options.enable_mem_pattern = True

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        options.log_severity_level = 3

        # ====================================================
        # CREATE SESSION
        # ====================================================

        try:

            session = ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=[
                    "CPUExecutionProvider",
                ],
            )

        except Exception as exc:

            raise RuntimeError(
                "Unable to initialize the Real-ESRGAN "
                f"ONNX Runtime session: {exc}"
            ) from exc

        # ====================================================
        # VERIFY EXECUTION PROVIDER
        # ====================================================

        providers = (
            session.get_providers()
        )

        if "CPUExecutionProvider" not in providers:

            raise RuntimeError(
                "ONNX Runtime CPUExecutionProvider "
                "is not available."
            )

        # ====================================================
        # STORE SHARED SESSION
        # ====================================================

        _session = session

        return _session
 
