import os
import threading

import onnxruntime as ort

from config import MODEL_PATH, CPU_THREADS


_session = None
_session_lock = threading.Lock()


def get_session():
    """
    Load the ONNX model once per process.

    Thread-safe lazy initialization prevents multiple workers/
    requests from loading duplicate model sessions.
    """

    global _session

    if _session is not None:
        return _session

    with _session_lock:

        if _session is not None:
            return _session

        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"ONNX model not found: {MODEL_PATH}"
            )

        opts = ort.SessionOptions()

        # Conservative settings for shared hosting.
        opts.intra_op_num_threads = max(
            1,
            CPU_THREADS,
        )

        opts.inter_op_num_threads = 1

        opts.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Memory settings.
        opts.enable_cpu_mem_arena = True
        opts.enable_mem_pattern = True

        # Avoid unnecessary logging.
        opts.log_severity_level = 3

        _session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=opts,
            providers=[
                "CPUExecutionProvider"
            ],
        )

        return _session