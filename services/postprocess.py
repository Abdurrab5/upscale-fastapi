import numpy as np

from PIL import Image


def tensor_to_image(
    output,
    alpha=None,
):

    output = np.asarray(
        output
    )

    if output.ndim == 4:

        output = output[0]

    if output.ndim != 3:

        raise ValueError(
            "Expected image tensor with 3 dimensions."
        )

    if output.shape[0] == 3:

        output = output.transpose(
            1,
            2,
            0,
        )

    elif output.shape[-1] == 3:

        pass

    else:

        raise ValueError(
            "Expected RGB image tensor."
        )

    output = np.clip(
        output,
        0.0,
        1.0,
    )

    output = (
        output
        * 255.0
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        output,
        mode="RGB",
    )

    if alpha is not None:

        alpha_image = Image.fromarray(
            alpha,
            mode="L",
        )

        try:

            if alpha_image.size != image.size:

                resized = alpha_image.resize(
                    image.size,
                    Image.Resampling.LANCZOS,
                )

                alpha_image.close()

                alpha_image = resized

            image.putalpha(
                alpha_image
            )

        finally:

            alpha_image.close()

    return image