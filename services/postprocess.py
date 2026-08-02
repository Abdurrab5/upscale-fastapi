import numpy as np
from PIL import Image


def tensor_to_image(output, alpha=None):

    output = output.squeeze(0)

    # CHW -> HWC
    output = output.transpose(1, 2, 0)

    output = np.clip(output, 0, 1)

    output = (output * 255).astype(np.uint8)

    image = Image.fromarray(output)

    if alpha is not None:

        alpha = Image.fromarray(alpha)

        alpha = alpha.resize(
            image.size,
            Image.LANCZOS
        )

        image.putalpha(alpha)

    return image