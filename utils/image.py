import os
import uuid


TEMP_DIR = "temp"

os.makedirs(TEMP_DIR, exist_ok=True)


def create_input_path(filename: str):

    ext = os.path.splitext(filename)[1].lower()

    return os.path.join(
        TEMP_DIR,
        f"{uuid.uuid4().hex}{ext}"
    )


def create_output_path():

    return os.path.join(
        TEMP_DIR,
        f"{uuid.uuid4().hex}.png"
    )


def cleanup(*files):

    for file in files:

        try:

            if file and os.path.exists(file):
                os.remove(file)

        except Exception:
            pass