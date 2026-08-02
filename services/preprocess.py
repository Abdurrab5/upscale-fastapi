import numpy as np
from PIL import Image, ImageOps


class ImageData:
    def __init__(self, tensor, alpha, original_size):
        self.tensor = tensor
        self.alpha = alpha
        self.original_size = original_size


def load_image(path: str) -> ImageData:
    """
    Loads image for ONNX inference.

    - Corrects EXIF orientation
    - Preserves alpha channel
    - Converts RGB
    - Converts to float32
    - Converts NCHW
    """

    img = Image.open(path)

    # Correct phone image orientation
    img = ImageOps.exif_transpose(img)

    alpha = None

    # Preserve alpha if present
    if img.mode == "RGBA":
        alpha = np.array(img.getchannel("A"))

    img = img.convert("RGB")

    width, height = img.size

    image = np.asarray(img, dtype=np.float32)

    # Normalize
    image /= 255.0

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    # CHW -> NCHW
    image = np.expand_dims(image, axis=0)

    return ImageData(
        tensor=image,
        alpha=alpha,
        original_size=(width, height)
    )