from dataclasses import dataclass


@dataclass(frozen=True)
class TargetResolution:

    width: int
    height: int

    source_width: int
    source_height: int

    quality: str

    scale: float

    model_scale: int

    strategy: str

    ai_passes: int

    needs_ai: bool

    @property
    def aspect_ratio(self) -> float:

        return (
            self.width
            / self.height
        )


# ============================================================
# ROUND
# ============================================================

def _round_dimension(
    value: float,
) -> int:

    value = max(
        2,
        int(round(value)),
    )

    if value % 2:
        value += 1

    return value


# ============================================================
# TARGET DIMENSIONS
# ============================================================

def _calculate_dimensions(
    source_width: int,
    source_height: int,
    longest_edge: int,
):

    source_longest = max(
        source_width,
        source_height,
    )

    scale = (
        longest_edge
        / source_longest
    )

    if source_width >= source_height:

        width = longest_edge

        height = _round_dimension(
            source_height * scale
        )

    else:

        height = longest_edge

        width = _round_dimension(
            source_width * scale
        )

    return (
        _round_dimension(width),
        _round_dimension(height),
    )


# ============================================================
# RESOLUTION ENGINE
# ============================================================

def resolve_target(
    source_width: int,
    source_height: int,
    quality: str,
) -> TargetResolution:

    if source_width <= 0:
        raise ValueError(
            "Invalid source image width."
        )

    if source_height <= 0:
        raise ValueError(
            "Invalid source image height."
        )

    quality = (
        str(quality or "")
        .strip()
        .lower()
    )

    from config import (
        QUALITY_TARGETS,
        MAX_OUTPUT_DIMENSION,
        MAX_OUTPUT_PIXELS,
        MODEL_SCALE,
    )

    if quality not in QUALITY_TARGETS:

        raise ValueError(
            "Please select 2K, 4K, or 8K."
        )

    longest_edge = min(
        QUALITY_TARGETS[quality],
        MAX_OUTPUT_DIMENSION,
    )

    target_width, target_height = (
        _calculate_dimensions(
            source_width,
            source_height,
            longest_edge,
        )
    )

    pixels = (
        target_width
        * target_height
    )

    if pixels > MAX_OUTPUT_PIXELS:

        reduction = (
            MAX_OUTPUT_PIXELS
            / pixels
        ) ** 0.5

        target_width = _round_dimension(
            target_width * reduction
        )

        target_height = _round_dimension(
            target_height * reduction
        )

    scale_x = (
        target_width
        / source_width
    )

    scale_y = (
        target_height
        / source_height
    )

    scale = (
        scale_x
        + scale_y
    ) / 2.0

    # ========================================================
    # PROCESSING STRATEGY
    # ========================================================

    if scale <= 1.0:

        strategy = "resize"
        ai_passes = 0
        needs_ai = False

    elif scale <= MODEL_SCALE:

        strategy = "ai_native"
        ai_passes = 1
        needs_ai = True

    else:

        strategy = "multi_stage_ai"

        # One x4 pass can cover up to 4x.
        # A second pass allows the pipeline to exceed 4x.
        ai_passes = 2

        needs_ai = True

    return TargetResolution(

        width=target_width,
        height=target_height,

        source_width=source_width,
        source_height=source_height,

        quality=quality,

        scale=scale,

        model_scale=MODEL_SCALE,

        strategy=strategy,

        ai_passes=ai_passes,

        needs_ai=needs_ai,
    )