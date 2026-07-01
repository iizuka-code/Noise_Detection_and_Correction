from __future__ import annotations

import json
import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dust_mask_repair.io import read_image, write_image
from dust_mask_repair.server import RepairWebHandler


ROOT = Path(__file__).resolve().parents[1]


def _artifact_path(name: str) -> Path:
    directory = ROOT / "test_outputs" / "server"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _add_disk(image: np.ndarray, cy: int, cx: int, radius: int, color: tuple[int, int, int]) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    image[disk] = np.asarray(color, dtype=np.uint8)


def test_repair_red_web_route_returns_repaired_image_and_generated_mask() -> None:
    normal = np.zeros((96, 96, 3), dtype=np.uint8)
    normal[:, :] = [90, 110, 130]
    normal[43:51, 43:51] = [255, 255, 255]
    red = np.zeros((96, 96, 3), dtype=np.uint8)
    red[:, :] = [5, 3, 4]
    _add_disk(red, 47, 47, 9, (48, 4, 5))
    _add_disk(red, 47, 47, 6, (220, 18, 24))
    _add_disk(red, 47, 47, 3, (255, 38, 42))

    normal_path = _artifact_path("normal.png")
    red_path = _artifact_path("red.png")
    write_image(normal_path, normal)
    write_image(red_path, red)

    server = ThreadingHTTPServer(("127.0.0.1", 0), RepairWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----DustMaskRepairRedBoundary"
        body = _multipart(
            boundary,
            files={
                "image": ("normal.png", normal_path.read_bytes(), "image/png"),
                "red_image": ("red.png", red_path.read_bytes(), "image/png"),
            },
            fields={
                "method": "hybrid",
                "mask_channel": "grayscale",
                "threshold": "0.5",
                "dilate_radius": "0",
                "feather_radius": "0",
                "strength": "1.0",
                "padding": "8",
                "detection_long_edge": "96",
                "max_red_area": "400",
                "max_red_dim": "32",
            },
        )
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/repair-red",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert payload["ok"] is True
        assert payload["red_highlight"]["detector_version"] == "red_highlight_v1"
        assert payload["metrics"]["max_abs_diff_outside_mask"] == 0.0
        assert payload["generated_mask_url"].endswith("/red_highlight/red_highlight_mask.png")
        assert payload["input_preview_url"].endswith("/input_preview.png")
        assert _get_status(port, payload["input_preview_url"]) == 200
        assert _get_status(port, payload["generated_mask_url"]) == 200
        assert _get_status(port, payload["repaired_url"]) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_white_dust_page_is_served() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), RepairWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/white_dust.html")
        response = conn.getresponse()
        body = response.read().decode("utf-8")

        assert response.status == 200
        assert "whiteDustForm" in body
        assert "黒/茶色地の検査画像" in body
        assert "補正対象画像" in body
        assert "/api/detect-white-dust" in body
        assert "/api/repair-white-dust" in body
    finally:
        server.shutdown()
        server.server_close()


def test_detect_white_dust_web_route_returns_mask_and_overlay() -> None:
    source = np.zeros((96, 128, 3), dtype=np.uint8)
    source[:, :] = [15, 16, 19]
    yy, xx = np.ogrid[:96, :128]
    dust = (yy - 48) ** 2 + (xx - 64) ** 2 <= 5 * 5
    source[dust] = [238, 232, 220]
    source[(yy - 60) ** 2 + (xx - 82) ** 2 <= 4 * 4] = [210, 120, 170]
    source_path = _artifact_path("white_dust_source.png")
    write_image(source_path, source)

    server = ThreadingHTTPServer(("127.0.0.1", 0), RepairWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----WhiteDustBoundary"
        body = _multipart(
            boundary,
            files={"source": ("source.png", source_path.read_bytes(), "image/png")},
            fields={
                "detection_long_edge": "128",
                "local_radius": "4",
                "max_area": "1200",
                "max_dim": "40",
                "max_thickness": "16",
            },
        )
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/detect-white-dust",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert payload["ok"] is True
        assert payload["white_dust"]["detector_version"] == "dust_on_dark_or_brown_v2"
        assert payload["white_dust"]["final_mask_pixels"] > 0
        assert payload["mask_output_mode"] == "legacy_plus_xmp"
        assert payload["xmp"]["format"] == "dust-mask-repair-xmp-v1"
        assert payload["xmp"]["output_mode"] == "legacy_plus_xmp"
        assert payload["xmp_url"].endswith("/white_dust/white_dust_mask.xmp")
        assert _get_status(port, payload["generated_mask_url"]) == 200
        assert _get_status(port, payload["manifest_url"]) == 200
        assert _get_status(port, payload["xmp_url"]) == 200
        assert _get_status(port, payload["overlay_url"]) == 200
        assert _get_status(port, payload["score_url"]) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_detect_white_dust_web_route_accepts_raw_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    source = np.zeros((96, 128, 3), dtype=np.uint16)
    source[:, :] = [4200, 4300, 5000]
    source[45:51, 61:67] = [61000, 59500, 57000]
    _install_fake_rawpy(monkeypatch, source)

    server = ThreadingHTTPServer(("127.0.0.1", 0), RepairWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----WhiteDustRawBoundary"
        body = _multipart(
            boundary,
            files={"source": ("source.dng", b"fake raw bytes", "image/x-adobe-dng")},
            fields={
                "detection_long_edge": "128",
                "local_radius": "4",
                "max_area": "1200",
                "max_dim": "40",
                "max_thickness": "16",
            },
        )
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/detect-white-dust",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert payload["ok"] is True
        assert payload["source_url"].endswith("/source.dng")
        assert payload["source_preview_url"].endswith("/white_dust/source_preview.png")
        assert payload["white_dust"]["source_shape"] == [96, 128]
        assert payload["white_dust"]["source_metadata"]["raw_suffix"] == ".dng"
        assert payload["white_dust"]["source_metadata"]["requested_output_bps"] == 8
        assert payload["white_dust"]["source_metadata"]["half_size"] is True
        assert payload["white_dust"]["raw_fast_proxy"] is True
        assert payload["white_dust"]["proxy_mask"] is True
        assert payload["white_dust"]["mask_shape"] == payload["white_dust"]["detection_shape"]
        assert payload["white_dust"]["final_mask_pixels"] > 0
        assert _get_status(port, payload["source_preview_url"]) == 200
        assert _get_status(port, payload["generated_mask_url"]) == 200
        assert _get_status(port, payload["manifest_url"]) == 200
        assert _get_status(port, payload["xmp_url"]) == 200
    finally:
        server.shutdown()
        server.server_close()


def test_repair_white_dust_web_route_fits_raw_mask_to_jpeg_target(monkeypatch: pytest.MonkeyPatch) -> None:
    source = np.zeros((96, 160, 3), dtype=np.uint16)
    source[:, :] = [4200, 4300, 5000]
    yy, xx = np.ogrid[:96, :160]
    source[(yy - 48) ** 2 + (xx - 80) ** 2 <= 6 * 6] = [61000, 59500, 57000]
    _install_fake_rawpy(monkeypatch, source)

    target = np.zeros((80, 80, 3), dtype=np.uint8)
    target[:, :] = [88, 104, 122]
    target[(np.ogrid[:80, :80][0] - 40) ** 2 + (np.ogrid[:80, :80][1] - 40) ** 2 <= 5 * 5] = [250, 250, 246]
    target_path = _artifact_path("white_dust_target.jpg")
    write_image(target_path, target)

    server = ThreadingHTTPServer(("127.0.0.1", 0), RepairWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----WhiteDustRepairBoundary"
        body = _multipart(
            boundary,
            files={
                "source": ("source.dng", b"fake raw bytes", "image/x-adobe-dng"),
                "image": ("target.jpg", target_path.read_bytes(), "image/jpeg"),
            },
            fields={
                "detection_long_edge": "160",
                "local_radius": "4",
                "max_area": "1200",
                "max_dim": "48",
                "max_thickness": "18",
                "method": "hybrid",
                "dilate_radius": "0",
                "feather_radius": "0",
                "padding": "10",
            },
        )
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/repair-white-dust",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert payload["ok"] is True
        assert payload["white_dust"]["raw_fast_proxy"] is True
        assert payload["mask_output_mode"] == "legacy_plus_xmp"
        assert payload["mask_mapping"]["mode"] == "center_crop_resize"
        assert payload["mask_mapping"]["target_shape"] == [80, 80]
        assert payload["metrics"]["max_abs_diff_outside_mask"] == 0.0
        assert payload["input_url"].endswith("/input.jpg")
        assert payload["repaired_url"].endswith("/repaired.jpg")
        assert payload["target_mask_overlay_url"].endswith("/white_dust/target_mask_overlay.png")
        assert payload["xmp_url"].endswith("/white_dust/white_dust_mask.xmp")
        assert payload["fitted_xmp_url"].endswith("/white_dust/fitted_repair_mask.xmp")
        assert _get_status(port, payload["generated_mask_url"]) == 200
        assert _get_status(port, payload["fitted_mask_url"]) == 200
        assert _get_status(port, payload["manifest_url"]) == 200
        assert _get_status(port, payload["fitted_xmp_url"]) == 200
        assert _get_status(port, payload["target_mask_overlay_url"]) == 200
        assert _get_status(port, payload["repaired_url"]) == 200

        repaired = read_image(_output_file(payload["repaired_url"])).pixels
        assert repaired.shape == target.shape
        assert float(np.mean(repaired[37:44, 37:44, :])) < 230.0
    finally:
        server.shutdown()
        server.server_close()


def test_repair_white_dust_web_route_accepts_arw_target(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_pixels = np.zeros((80, 80, 3), dtype=np.uint16)
    raw_pixels[:, :] = [22000, 26000, 30000]
    yy, xx = np.ogrid[:80, :80]
    raw_pixels[(yy - 40) ** 2 + (xx - 40) ** 2 <= 5 * 5] = [62000, 61000, 59000]
    _install_fake_rawpy(monkeypatch, raw_pixels)

    server = ThreadingHTTPServer(("127.0.0.1", 0), RepairWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----WhiteDustRepairArwTargetBoundary"
        body = _multipart(
            boundary,
            files={
                "source": ("source.dng", b"fake raw bytes", "image/x-adobe-dng"),
                "image": ("target.arw", b"fake arw bytes", "image/x-sony-arw"),
            },
            fields={
                "detection_long_edge": "80",
                "local_radius": "4",
                "max_area": "1200",
                "max_dim": "48",
                "max_thickness": "18",
                "method": "linear",
                "dilate_radius": "0",
                "feather_radius": "0",
            },
        )
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/repair-white-dust",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert payload["ok"] is True
        assert payload["input_original_url"].endswith("/input.arw")
        assert payload["input_url"].endswith("/input_preview.png")
        assert payload["repaired_url"].endswith("/repaired.png")
        assert _get_status(port, payload["input_preview_url"]) == 200
        assert _get_status(port, payload["repaired_url"]) == 200

        repaired = read_image(_output_file(payload["repaired_url"])).pixels
        assert repaired.shape == raw_pixels.shape
        assert repaired.dtype == np.uint16
    finally:
        server.shutdown()
        server.server_close()


def test_repair_white_dust_web_route_rejects_unsupported_target_extension() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), RepairWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----WhiteDustRepairRejectUnsupportedBoundary"
        body = _multipart(
            boundary,
            files={
                "source": ("source.dng", b"fake raw bytes", "image/x-adobe-dng"),
                "image": ("target.txt", b"not an image", "text/plain"),
            },
            fields={},
        )
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/repair-white-dust",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 400
        assert payload["ok"] is False
        assert "Unsupported uploaded file extension" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_repair_red_web_route_rejects_size_mismatch() -> None:
    normal_path = _artifact_path("normal_mismatch.png")
    red_path = _artifact_path("red_mismatch.png")
    write_image(normal_path, np.zeros((32, 32, 3), dtype=np.uint8))
    write_image(red_path, np.zeros((30, 32, 3), dtype=np.uint8))

    server = ThreadingHTTPServer(("127.0.0.1", 0), RepairWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----DustMaskRepairRedMismatchBoundary"
        body = _multipart(
            boundary,
            files={
                "image": ("normal.png", normal_path.read_bytes(), "image/png"),
                "red_image": ("red.png", red_path.read_bytes(), "image/png"),
            },
            fields={"detection_long_edge": "32"},
        )
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/repair-red",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 400
        assert payload["ok"] is False
        assert "dimensions differ" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()


def _get_status(port: int, path: str) -> int:
    conn = HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    response = conn.getresponse()
    response.read()
    return int(response.status)


def _output_file(path: str) -> Path:
    return ROOT / "web_outputs" / path.removeprefix("/outputs/")


def _multipart(
    boundary: str,
    files: dict[str, tuple[str, bytes, str]],
    fields: dict[str, str],
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    for name, (filename, data, content_type) in files.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


class _FakeRawFile:
    def __init__(self, pixels: np.ndarray) -> None:
        self._pixels = pixels

    def __enter__(self) -> "_FakeRawFile":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        return None

    def postprocess(self, **_kwargs: object) -> np.ndarray:
        return self._pixels


def _install_fake_rawpy(monkeypatch: pytest.MonkeyPatch, pixels: np.ndarray) -> None:
    fake_rawpy = SimpleNamespace(
        ColorSpace=SimpleNamespace(sRGB="srgb"),
        imread=lambda _path: _FakeRawFile(pixels),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)
