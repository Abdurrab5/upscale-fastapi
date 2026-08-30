
import numpy as np

from PIL import Image


# ============================================================
# TENSOR → PIL IMAGE
# ============================================================

def tensor_to_image(
    output,
    alpha=None,
):
    """
    Convert an ONNX/NumPy output tensor to a PIL image.

    Expected model output:

        [1, C, H, W]

    or:

        [C, H, W]

    Supported color output:

        RGB
        RGBA

    Alpha is preserved separately when supplied.
    """

    if output is None:
        raise ValueError(
            "Model output is empty."
        )

    # ========================================================
    # NUMPY CONVERSION
    # ========================================================

    output = np.asarray(
        output
    )

    # ========================================================
    # REMOVE BATCH DIMENSION
    # ========================================================

    if output.ndim == 4:

        if output.shape[0] != 1:
            raise ValueError(
                "Expected a single image in the model output."
            )

        output = output[0]

    # ========================================================
    # CHW → HWC
    # ========================================================

    if output.ndim != 3:
        raise ValueError(
            "Expected image tensor with shape CHW or HWC."
        )

    # Real-ESRGAN / ONNX output is normally CHW.
    #
    # Detect channel-first layout safely.

    if output.shape[0] in (1, 3, 4):

        output = output.transpose(
            1,
            2,
            0,
        )

    elif output.shape[-1] not in (1, 3, 4):

        raise ValueError(
            "Unsupported model output shape."
        )

    # ========================================================
    # NORMALIZE OUTPUT
    # ========================================================

    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    output = np.clip(
        output,
        0.0,
        1.0,
    )

    output = (
        output * 255.0
    ).round().astype(
        np.uint8
    )

    # ========================================================
    # CREATE RGB IMAGE
    # ========================================================

    if output.shape[2] == 1:

        output = np.repeat(
            output,
            3,
            axis=2,
        )

    if output.shape[2] == 4:

        image = Image.fromarray(
            output,
            mode="RGBA",
        )

    else:

        image = Image.fromarray(
            output,
            mode="RGB",
        )

    # ========================================================
    # PRESERVE ALPHA
    # ========================================================

    if alpha is not None:

        alpha = np.asarray(
            alpha
        )

        # ----------------------------------------------------
        # Remove unnecessary dimensions
        # ----------------------------------------------------

        alpha = np.squeeze(
            alpha
        )

        if alpha.ndim != 2:
            raise ValueError(
                "Alpha channel must be a 2D array."
            )

        # ----------------------------------------------------
        # Normalize alpha
        # ----------------------------------------------------

        if (
            np.issubdtype(
                alpha.dtype,
                np.floating,
            )
            and alpha.max(initial=0) <= 1.0
        ):

            alpha = (
                np.clip(
                    alpha,
                    0.0,
                    1.0,
                )
                * 255.0
            ).round().astype(
                np.uint8
            )

        else:

            alpha = np.clip(
                alpha,
                0,
                255,
            ).astype(
                np.uint8
            )

        # ----------------------------------------------------
        # Create alpha image
        # ----------------------------------------------------

        alpha_image = Image.fromarray(
            alpha,
            mode="L",
        )

        # ----------------------------------------------------
        # Resize alpha to final AI output
        # ----------------------------------------------------

        if alpha_image.size != image.size:

            alpha_image = alpha_image.resize(
                image.size,
                Image.Resampling.LANCZOS,
            )

        # ----------------------------------------------------
        # Ensure RGB image receives alpha correctly
        # ----------------------------------------------------

        if image.mode != "RGBA":

            image = image.convert(
                "RGBA"
            )

        image.putalpha(
            alpha_image
        )

        alpha_image.close()

    return image
 
