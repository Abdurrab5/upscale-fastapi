import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "realesr-general-x4v3.onnx"
)

SCALE = 4

# tile sizes
TILE_SMALL = 128
TILE_MEDIUM = 256
TILE_LARGE = 512

# overlap between tiles
TILE_PAD = 16
 
# Thread count
CPU_THREADS = min(os.cpu_count() or 4, 8)

# Output
OUTPUT_FORMAT = "PNG"