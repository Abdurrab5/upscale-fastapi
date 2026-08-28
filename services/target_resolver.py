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

    model_input_width: int

    model_input_height: int

    @property
    def aspect_ratio(self) -> float:

        return (
            self.width
            / self.height
        )


# ============================================================
# DIMENSION ROUNDING
# ============================================================

def _round_dimension(
    value: float,
) -> int:

    result = max(
        2,
        int(round(value)),
    )

    if result % 2:
        result += 1

    return result


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

        height = (
            source_height
            * scale
        )

    else:

        height = longest_edge

        width = (
            source_width
            * scale
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
        QUALITY_MODEL,
        FAST_X2_SCALE,
        REAL_ESRGAN_SCALE,
        MAX_OUTPUT_DIMENSION,
        MAX_OUTPUT_PIXELS,
    )

    if quality not in QUALITY_TARGETS:

        raise ValueError(
            "Please select HD, 2K, or 4K."
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

    # --------------------------------------------------------
    # Pixel safety
    # --------------------------------------------------------

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
            target_width
            * reduction
        )

        target_height = _round_dimension(
            target_height
            * reduction
        )

    # --------------------------------------------------------
    # Final scale
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Model routing
    # --------------------------------------------------------

    model = QUALITY_MODEL[
        quality
    ]

    # --------------------------------------------------------
    # Downscale
    # --------------------------------------------------------

    if scale <= 1.0:

        return TargetResolution(
            width=target_width,
            height=target_height,

            source_width=source_width,
            source_height=source_height,

            quality=quality,

            scale=scale,

            model_scale=1,

            strategy="resize",

            ai_passes=0,

            needs_ai=False,

            model_input_width=source_width,
            model_input_height=source_height,
        )

    # --------------------------------------------------------
    # Fast x2
    # --------------------------------------------------------

    if model == "fast_x2":

        return TargetResolution(
            width=target_width,
            height=target_height,

            source_width=source_width,
            source_height=source_height,

            quality=quality,

            scale=scale,

            model_scale=FAST_X2_SCALE,

            strategy="fast_x2",

            ai_passes=1,

            needs_ai=True,

            model_input_width=source_width,
            model_input_height=source_height,
        )

    # --------------------------------------------------------
    # Real-ESRGAN x4
    # --------------------------------------------------------

    return TargetResolution(
        width=target_width,
        height=target_height,

        source_width=source_width,
        source_height=source_height,

        quality=quality,

        scale=scale,

        model_scale=REAL_ESRGAN_SCALE,

        strategy="best_x4",

        ai_passes=1,

        needs_ai=True,

        model_input_width=source_width,
        model_input_height=source_height,
    )