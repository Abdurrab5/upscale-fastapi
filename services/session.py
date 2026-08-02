import onnxruntime as ort

from config import MODEL_PATH
from config import CPU_THREADS

_session = None


def get_session():
    global _session

    if _session is not None:
        return _session

    opts = ort.SessionOptions()

    opts.intra_op_num_threads = CPU_THREADS
    opts.inter_op_num_threads = 1

    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    _session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )

    return _session