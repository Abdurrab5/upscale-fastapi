
from dataclasses import dataclass


# ============================================================
# TARGET RESOLUTION
# ============================================================

@dataclass(frozen=True)
class TargetResolution:
    """
    Immutable description of the requested final resolution.

    Quality modes:

        HD  → 2048px longest edge
        2K  → 2048px longest edge
        4K  → 4096px longest edge

    The source aspect ratio is preserved.
    """

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
# ROUND DIMENSION
# ============================================================

def _round_dimension(
    value: float,
) -> int:
    """
    Round a dimension to a positive even pixel value.
    """

    value = max(
        2,
        int(round(value)),
    )

    if value % 2 != 0:
        value += 1

    return value


# ============================================================
# CALCULATE DIMENSIONS
# ============================================================

def _calculate_dimensions(
    source_width: int,
    source_height: int,
    longest_edge: int,
):
    """
    Calculate final dimensions while preserving aspect ratio.

    The requested longest edge becomes the final longest edge,
    subject to even-pixel rounding.

    Examples:

        826 × 1062

        HD → approximately 1594 × 2048
        2K → approximately 1594 × 2048
        4K → approximately 3186 × 4096
    """

    source_longest = max(
        source_width,
        source_height,
    )

    if source_longest <= 0:
        raise ValueError(
            "Invalid source image dimensions."
        )

    if longest_edge <= 0:
        raise ValueError(
            "Invalid target resolution."
        )

    scale = (
        longest_edge
        / source_longest
    )

    width = _round_dimension(
        source_width * scale
    )

    height = _round_dimension(
        source_height * scale
    )

    return (
        width,
        height,
    )


# ============================================================
# SCALE TO SAFETY LIMIT
# ============================================================

def _fit_to_limits(
    width: int,
    height: int,
    max_dimension: int,
    max_pixels: int,
):
    """
    Reduce dimensions proportionally so the final output
    stays within both dimension and pixel-count limits.
    """

    width = int(width)
    height = int(height)

    # ========================================================
    # DIMENSION LIMIT
    # ========================================================

    actual_longest = max(
        width,
        height,
    )

    if actual_longest > max_dimension:

        reduction = (
            max_dimension
            / actual_longest
        )

        width = _round_dimension(
            width * reduction
        )

        height = _round_dimension(
            height * reduction
        )

    # ========================================================
    # PIXEL LIMIT
    # ========================================================

    pixels = (
        width
        * height
    )

    if pixels > max_pixels:

        reduction = (
            max_pixels
            / pixels
        ) ** 0.5

        width = _round_dimension(
            width * reduction
        )

        height = _round_dimension(
            height * reduction
        )

    return (
        width,
        height,
    )


# ============================================================
# RESOLUTION ENGINE
# ============================================================

def resolve_target(
    source_width: int,
    source_height: int,
    quality: str,
) -> TargetResolution:
    """
    Resolve the final output resolution.

    Public quality modes:

        hd → 2048px longest edge
        2k → 2048px longest edge
        4k → 4096px longest edge

    Aspect ratio is preserved.

    Real-ESRGAN General x4 v3 is used for AI enlargement.

    Important:

        Quality represents the FINAL output resolution.
        It does not represent source × 2 / × 4 / × 8.
    """

    # ========================================================
    # SOURCE VALIDATION
    # ========================================================

    source_width = int(
        source_width
    )

    source_height = int(
        source_height
    )

    if source_width <= 0:
        raise ValueError(
            "Invalid source image width."
        )

    if source_height <= 0:
        raise ValueError(
            "Invalid source image height."
        )

    # ========================================================
    # CONFIG
    # ========================================================

    from config import (
        QUALITY_TARGETS,
        MAX_OUTPUT_DIMENSION,
        MAX_OUTPUT_PIXELS,
        MODEL_SCALE,
    )

    # ========================================================
    # QUALITY NORMALIZATION
    # ========================================================

    quality = (
        str(quality or "")
        .strip()
        .lower()
    )

    # HD is intentionally an alias of 2K.
    #
    # Both produce a 2048px longest edge, but we preserve
    # the requested public quality name in the result.

    if quality == "2048":
        quality = "2k"

    if quality not in QUALITY_TARGETS:

        raise ValueError(
            "Please select HD, 2K, or 4K."
        )

    # ========================================================
    # REQUESTED LONGEST EDGE
    # ========================================================

    requested_longest_edge = int(
        QUALITY_TARGETS[
            quality
        ]
    )

    if requested_longest_edge <= 0:

        raise ValueError(
            "Invalid configured target resolution."
        )

    # Never allow configuration to exceed the absolute
    # server-side safety limit.

    longest_edge = min(
        requested_longest_edge,
        MAX_OUTPUT_DIMENSION,
    )

    # ========================================================
    # CALCULATE TARGET
    # ========================================================

    target_width, target_height = (
        _calculate_dimensions(
            source_width,
            source_height,
            longest_edge,
        )
    )

    # ========================================================
    # APPLY SAFETY LIMITS
    # ========================================================

    target_width, target_height = (
        _fit_to_limits(
            target_width,
            target_height,
            MAX_OUTPUT_DIMENSION,
            MAX_OUTPUT_PIXELS,
        )
    )

    # ========================================================
    # FINAL SAFETY CHECK
    # ========================================================

    if (
        target_width <= 0
        or target_height <= 0
    ):

        raise ValueError(
            "Unable to calculate a valid output resolution."
        )

    if (
        target_width > MAX_OUTPUT_DIMENSION
        or target_height > MAX_OUTPUT_DIMENSION
    ):

        raise ValueError(
            "Requested output dimensions exceed "
            "the maximum supported resolution."
        )

    if (
        target_width
        * target_height
        > MAX_OUTPUT_PIXELS
    ):

        raise ValueError(
            "Requested output resolution exceeds "
            "the maximum supported pixel count."
        )

    # ========================================================
    # ACTUAL SCALE
    # ========================================================

    scale_x = (
        target_width
        / source_width
    )

    scale_y = (
        target_height
        / source_height
    )

    # Because dimensions are independently rounded to even
    # values, calculate the actual average scale.

    scale = (
        scale_x
        + scale_y
    ) / 2.0

    # ========================================================
    # PROCESSING STRATEGY
    # ========================================================

    if scale <= 1.0:

        # ----------------------------------------------------
        # Source already meets/exceeds requested resolution.
        #
        # No AI enlargement is required.
        # The processing layer can resize down while
        # preserving the requested final dimensions.
        # ----------------------------------------------------

        strategy = "resize"

        ai_passes = 0

        needs_ai = False

    else:

        # ----------------------------------------------------
        # AI enlargement required.
        #
        # The Real-ESRGAN model is x4 internally. The inference
        # engine is responsible for producing the requested
        # final dimensions.
        #
        # Do not expose x2/x4/x8 as public quality modes.
        # ----------------------------------------------------

        strategy = "ai"

        ai_passes = 1

        needs_ai = True

    # ========================================================
    # RESULT
    # ========================================================

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
 
