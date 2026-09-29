# CPU and memory tuning

The service keeps conservative defaults and reads performance settings from
the environment when `config.py` loads. Restart the process after changing
them.

| Setting | Default | Valid values | Purpose |
| --- | ---: | --- | --- |
| `UPSCALE_CPU_THREADS` | `2` | 1–4, capped by available CPUs | ONNX intra-op CPU threads |
| `UPSCALE_TILE_SMALL` | `128` | 64–512, greater than twice tile padding | Tile size for sources up to 2 MP |
| `UPSCALE_TILE_MEDIUM` | `192` | 64–512, greater than twice tile padding | Tile size for sources up to 6 MP |
| `UPSCALE_TILE_LARGE` | `256` | 64–512, greater than twice tile padding | Tile size for larger sources |
| `UPSCALE_TILE_PAD` | `16` | 1–64 | Source pixels cropped from tile overlaps |
| `UPSCALE_PNG_COMPRESS_LEVEL` | `6` | 0–9 | Lossless PNG compression effort |
| `ONNX_ENABLE_CPU_MEM_ARENA` | `true` | true/false | ONNX CPU memory arena |
| `ONNX_ENABLE_MEM_PATTERN` | `true` | true/false | ONNX memory pattern optimization |

Inference concurrency remains one job. The shared ONNX session uses the CPU
provider, sequential execution, and full graph optimization. Progress JSON is
written atomically, throttled to meaningful percentage changes or 0.5-second
intervals, and stale files are removed after 24 hours.

To benchmark an existing local image without adding it to the repository:

```bash
python scripts/benchmark_upscale.py /path/to/image.jpg --quality 4k --format png
```

The script reports dimensions, tile count, elapsed time, output size, and peak
RSS. Stage timings are logged by the service. For example, benchmark a larger
tile setting in a fresh process with `UPSCALE_TILE_SMALL=256` prefixed to the
command. Larger tiles reduce model calls but increase peak memory; benchmark
with representative inputs before changing production values.
