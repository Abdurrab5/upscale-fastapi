from services.session import get_session
from services.tile_processor import merge_tiles


class InferenceEngine:

    def __init__(self):
        self.session = get_session()

    def upscale(
        self,
        tensor,
        progress_callback=None
    ):
        return merge_tiles(
            self.session,
            tensor,
            progress_callback
        )


_engine = InferenceEngine()


def get_engine():
    return _engine