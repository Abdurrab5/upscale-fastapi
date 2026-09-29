#!/usr/bin/env python3
"""Run a local upscale benchmark against developer-supplied images.

The detailed stage timings are emitted by services.upscale_service as
UPSCALE_METRICS log lines. This script adds input/output dimensions, file size,
tile count, wall time, and process peak RSS for easy side-by-side runs.
"""

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.preprocess import load_image
from services.target_resolver import resolve_target
from services.upscale_service import upscale_image
from services.tile_processor import choose_tile_size, tile_count
from utils.image import create_output_path, cleanup
from utils.metrics import peak_rss_mb
from utils.progress import remove_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Input JPG, PNG, or WebP paths")
    parser.add_argument("--quality", choices=("hd", "2k", "4k"), default="4k")
    parser.add_argument("--format", choices=("png", "jpeg", "webp"), default="png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for path in args.images:
        with Image.open(path) as source:
            source_size = source.size
        target = resolve_target(*source_size, args.quality)
        output_path = create_output_path(args.format)
        job_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            upscale_image(path, output_path, job_id, target.quality, target.width, target.height, args.format)
            elapsed = time.perf_counter() - started
            with Image.open(output_path) as result:
                output_size = result.size
            rss = peak_rss_mb()
            tiles = tile_count(*source_size) if target.needs_ai else 0
            tile = choose_tile_size(*source_size)
            rss_value = f"{rss:.1f}" if rss is not None else "unavailable"
            print(
                f"BENCHMARK image={os.path.basename(path)} source={source_size[0]}x{source_size[1]} "
                f"target={output_size[0]}x{output_size[1]} quality={target.quality} tile={tile} "
                f"tiles={tiles} total_s={elapsed:.3f} output_bytes={os.path.getsize(output_path)} peak_rss_mb={rss_value}"
            )
        finally:
            cleanup(output_path)
            remove_progress(job_id)


if __name__ == "__main__":
    main()
