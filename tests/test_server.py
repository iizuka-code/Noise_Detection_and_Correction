from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from dust_mask_repair.io import write_image
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
        assert _get_status(port, payload["generated_mask_url"]) == 200
        assert _get_status(port, payload["repaired_url"]) == 200
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
