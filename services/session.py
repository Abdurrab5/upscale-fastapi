import os
import threading

import onnxruntime as ort

from config import (
    CPU_THREADS,
    MODEL_PATH,
)


_session = None
_session_lock = threading.Lock()


def get_session():

    global _session

    if _session is not None:
        return _session

    with _session_lock:

        if _session is not None:
            return _session

        if not os.path.isfile(
            MODEL_PATH
        ):

            raise FileNotFoundError(
                f"ONNX model not found: {MODEL_PATH}"
            )

        options = ort.SessionOptions()

        options.intra_op_num_threads = max(
            1,
            CPU_THREADS,
        )

        options.inter_op_num_threads = 1

        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True

        options.log_severity_level = 3

        _session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=options,
            providers=[
                "CPUExecutionProvider",
            ],
        )

        return _session