import io
import os
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import routes.upscale as upscale_route
from main import app
from services.inference import InferenceEngine
import services.tile_processor as tile_processor
import services.session as session_module
from services.preprocess import load_image
from services.tile_processor import generate_tiles, run_ai_pass, tile_count
from services.target_resolver import resolve_target
from services.output_writer import UpscaleOutputWriter
from utils.image import create_output_path
import utils.progress as progress_store


client = TestClient(app)


def image_bytes(fmt="PNG", mode="RGB", size=(17, 11)):
    image = Image.new(mode, size, (30, 80, 140, 64) if mode == "RGBA" else (30, 80, 140))
    output = io.BytesIO()
    image.save(output, format=fmt)
    image.close()
    return output.getvalue()


def test_output_path_extensions():
    assert create_output_path("png").endswith(".png")
    assert create_output_path("jpeg").endswith(".jpg")
    assert create_output_path("webp").endswith(".webp")


def test_target_resolution_and_direct_resize_plan():
    hd = resolve_target(200, 100, "hd")
    legacy_2k = resolve_target(200, 100, "2k")
    four_k = resolve_target(200, 100, "4k")
    downscale = resolve_target(5000, 2500, "hd")
    assert max(hd.width, hd.height) == 2048
    assert max(four_k.width, four_k.height) == 4096
    assert (legacy_2k.width, legacy_2k.height) == (hd.width, hd.height)
    assert abs(hd.width / hd.height - 2.0) < 0.002
    assert downscale.strategy == "resize"
    assert downscale.ai_passes == 0
    portrait = resolve_target(826, 1062, "4k")
    assert max(portrait.width, portrait.height) == 4096
    assert abs(portrait.width / portrait.height - 826 / 1062) < 0.001


def test_supported_output_formats_and_png_alpha(monkeypatch, tmp_path):
    monkeypatch.setattr(progress_store, "PROGRESS_DIR", str(tmp_path))
    output_paths = []

    def fake_upscale(input_path, output_path, job_id, quality, width, height, output_format):
        output_paths.append((output_path, output_format))
        mode = "RGBA" if output_format == "png" else "RGB"
        fill = (20, 40, 60, 73) if mode == "RGBA" else (20, 40, 60)
        Image.new(mode, (width, height), fill).save(output_path, format=output_format.upper())

    monkeypatch.setattr(upscale_route, "upscale_image", fake_upscale)
    formats = [
        ("JPEG", "image/jpeg", "photo.jpg", "jpeg", "image/jpeg", ".jpg"),
        ("JPEG", "image/jpeg", "photo.jpg", "png", "image/png", ".png"),
        ("PNG", "image/png", "photo.png", "png", "image/png", ".png"),
        ("PNG", "image/png", "photo.png", "webp", "image/webp", ".webp"),
        ("WEBP", "image/webp", "photo.webp", "webp", "image/webp", ".webp"),
    ]
    for source_fmt, mime, name, output_fmt, expected_mime, extension in formats:
        response = client.post(
            "/upscale",
            data={"job_id": uuid.uuid4().hex, "quality": "hd", "output_format": output_fmt},
            files={"file": (name, image_bytes(source_fmt), mime)},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith(expected_mime)
        assert response.headers["content-disposition"].endswith(extension + '"')
        assert response.headers.get("content-encoding") != "gzip"
        with Image.open(io.BytesIO(response.content)) as result:
            assert result.size == (2048, 1326)

    alpha_response = client.post(
        "/upscale",
        data={"job_id": uuid.uuid4().hex, "quality": "hd", "output_format": "png"},
        files={"file": ("alpha.png", image_bytes("PNG", "RGBA"), "image/png")},
    )
    assert alpha_response.status_code == 200
    with Image.open(io.BytesIO(alpha_response.content)) as result:
        assert result.mode == "RGBA"
        assert result.getchannel("A").getextrema() == (73, 73)

    assert [path.rsplit(".", 1)[-1] for path, _ in output_paths] == ["jpg", "png", "png", "webp", "webp", "png"]
    assert all(not os.path.exists(path) for path, _ in output_paths)


def test_writer_preserves_alpha(tmp_path):
    path = str(tmp_path / "alpha.png")
    writer = UpscaleOutputWriter(8, 6, path, "png")
    writer.create()
    writer.write_tile(np.full((6, 8, 3), 120, dtype=np.uint8), 0, 0)
    writer.finalize(alpha=np.full((3, 4), 73, dtype=np.uint8))
    writer.close()
    with Image.open(path) as result:
        assert result.mode == "RGBA"
        assert result.getchannel("A").getextrema() == (73, 73)


@pytest.mark.parametrize("image_format, extension", [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp")])
def test_writer_encodes_supported_formats(tmp_path, image_format, extension):
    path = str(tmp_path / f"output{extension}")
    output_format = "jpeg" if image_format == "JPEG" else image_format.lower()
    writer = UpscaleOutputWriter(12, 8, path, output_format)
    writer.create()
    writer.write_tile(np.full((8, 12, 3), 100, dtype=np.uint8), 0, 0)
    writer.finalize()
    writer.close()
    with Image.open(path) as result:
        assert result.format == image_format
        assert result.size == (12, 8)


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (13, 7), (10, 20, 30))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    image.close()
    prepared = load_image(str(path))
    assert prepared.original_size == (7, 13)
    assert prepared.tensor.shape == (13, 7, 3)


def test_tile_generator_and_direct_writer(tmp_path):
    source = np.full((70, 100, 3), 90, dtype=np.uint8)
    assert tile_count(100, 70) == 1
    assert len(list(generate_tiles(source))) == 1
    multi_tile = np.zeros((260, 300, 3), dtype=np.uint8)
    assert tile_count(300, 260) == len(list(generate_tiles(multi_tile)))

    class FakeSession:
        def get_inputs(self):
            return [type("Input", (), {"name": "input"})()]

        def run(self, _outputs, inputs):
            value = inputs["input"]
            return [np.repeat(np.repeat(value, 4, axis=2), 4, axis=3)]

    output_path = str(tmp_path / "tile.png")
    writer = UpscaleOutputWriter(200, 140, output_path, "png")
    writer.create()
    result = run_ai_pass(FakeSession(), source, 200, 140, output_writer=writer)
    assert result is None
    writer.finalize()
    writer.close()
    with Image.open(output_path) as output:
        assert output.size == (200, 140)
        assert output.getpixel((100, 70)) == (90, 90, 90)


@pytest.mark.parametrize(
    "width,height,tile_size,expected_count",
    [
        (826, 1062, 128, 99),
        # The checked-in iterator stops at the first tile covering each edge,
        # so 192/pad-16 produces 5 columns x 7 rows, not 42 tiles.
        (826, 1062, 192, 35),
        (826, 1062, 256, 20),
        (70, 100, 128, 1),  # Smaller than one tile.
        (192, 192, 192, 1),  # Exactly one tile.
        (352, 352, 192, 4),  # Exactly tile size plus one stride.
        (353, 353, 192, 9),  # One pixel beyond that boundary.
        (826, 1062, 192, 35),  # Portrait.
        (1062, 826, 192, 35),  # Landscape.
        (826, 826, 192, 25),  # Square.
    ],
)
def test_tile_count_matches_lazy_iterator(
    monkeypatch, width, height, tile_size, expected_count
):
    monkeypatch.setattr(tile_processor, "TILE_SMALL", tile_size)
    monkeypatch.setattr(tile_processor, "TILE_PAD", 16)
    source = np.zeros((height, width, 3), dtype=np.uint8)

    generated_count = sum(1 for _ in generate_tiles(source))

    assert generated_count == expected_count
    assert tile_count(width, height) == generated_count


def test_busy_inference_lock_is_preserved():
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.job_lock = __import__("threading").Lock()
    engine.job_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="currently busy"):
            engine.upscale(np.zeros((1, 1, 3), dtype=np.uint8), 10, 10)
    finally:
        engine.job_lock.release()


def test_onnx_session_is_shared(monkeypatch):
    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    created = []

    def make_session(*args, **kwargs):
        instance = FakeSession()
        created.append(instance)
        return instance

    monkeypatch.setattr(session_module, "_session", None)
    monkeypatch.setattr(session_module.ort, "InferenceSession", make_session)
    assert session_module.get_session() is session_module.get_session()
    assert len(created) == 1


def test_progress_updates_are_throttled_and_atomic(monkeypatch, tmp_path):
    monkeypatch.setattr(progress_store, "PROGRESS_DIR", str(tmp_path))
    progress_store._LAST_WRITES.clear()
    progress_store.update_progress("job12345", 10, "step")
    progress_store.update_progress("job12345", 11, "minor step")
    assert progress_store.get_progress("job12345")["percent"] == 10
    progress_store.update_progress("job12345", 12, "next step")
    assert progress_store.get_progress("job12345")["percent"] == 12
    progress_store.update_progress("job12345", 100, "done", "completed")
    assert progress_store.get_progress("job12345")["status"] == "completed"


def test_validation_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(progress_store, "PROGRESS_DIR", str(tmp_path))
    valid = image_bytes()
    invalid_job = client.post("/upscale", data={"job_id": "bad", "quality": "hd"}, files={"file": ("x.png", valid, "image/png")})
    assert invalid_job.status_code == 400

    bad_extension = client.post("/upscale", data={"job_id": uuid.uuid4().hex}, files={"file": ("x.gif", valid, "image/png")})
    assert bad_extension.status_code == 415

    bad_mime = client.post("/upscale", data={"job_id": uuid.uuid4().hex}, files={"file": ("x.png", valid, "application/octet-stream")})
    assert bad_mime.status_code == 415

    bad_image = client.post("/upscale", data={"job_id": uuid.uuid4().hex}, files={"file": ("x.png", b"not an image", "image/png")})
    assert bad_image.status_code == 400

    bad_edge = client.post("/upscale", data={"job_id": uuid.uuid4().hex, "target_longest_edge": "3000"}, files={"file": ("x.png", valid, "image/png")})
    assert bad_edge.status_code == 400

    monkeypatch.setattr(upscale_route, "MAX_IMAGE_PIXELS", 10)
    too_many_pixels = client.post("/upscale", data={"job_id": uuid.uuid4().hex}, files={"file": ("x.png", valid, "image/png")})
    assert too_many_pixels.status_code == 400

    too_large = client.post("/upscale", data={"job_id": uuid.uuid4().hex}, files={"file": ("x.png", b"0" * (10 * 1024 * 1024 + 1), "image/png")})
    assert too_large.status_code == 413


def test_existing_get_endpoints():
    assert client.get("/").json()["service"] == "Xhunta AI Image Upscaler"
    assert client.get("/health").json()["status"] == "healthy"
    assert "output_formats" in client.get("/capabilities").json()
    assert client.get("/progress/job12345").json()["status"] == "waiting"
