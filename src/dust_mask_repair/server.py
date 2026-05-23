from __future__ import annotations

import argparse
import json
import shutil
import uuid
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import cgi

from .config import MASK_CHANNELS, REPAIR_METHODS, RepairConfig
from .io import read_image, write_image
from .repair import repair_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
OUTPUT_ROOT = PROJECT_ROOT / "web_outputs"


class RepairWebHandler(BaseHTTPRequestHandler):
    server_version = "DustMaskRepairWeb/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path.endswith(".html"):
            rel = unquote(path.lstrip("/"))
            candidate = (WEB_ROOT / rel).resolve()
            if not _is_relative_to(candidate, WEB_ROOT.resolve()) or not candidate.is_file():
                self._send_error(HTTPStatus.NOT_FOUND, "html file not found")
                return
            self._send_file(candidate, _content_type(candidate))
            return
        if path.startswith("/outputs/"):
            rel = unquote(path.removeprefix("/outputs/"))
            candidate = (OUTPUT_ROOT / rel).resolve()
            if not _is_relative_to(candidate, OUTPUT_ROOT.resolve()) or not candidate.is_file():
                self._send_error(HTTPStatus.NOT_FOUND, "output file not found")
                return
            self._send_file(candidate, _content_type(candidate))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/repair":
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            response = self._handle_repair()
        except Exception as exc:  # noqa: BLE001 - return failures to the local UI.
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(response)

    def _handle_repair(self) -> dict[str, object]:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        image_field = _required_file(form, "image")
        mask_field = _required_file(form, "mask")
        run_id = uuid.uuid4().hex[:12]
        run_dir = OUTPUT_ROOT / run_id
        debug_dir = run_dir / "debug"
        run_dir.mkdir(parents=True, exist_ok=True)

        image_path = _save_upload(image_field, run_dir, "input", ".png")
        mask_path = _save_upload(mask_field, run_dir, "mask", ".png")

        cfg = RepairConfig(
            method=_form_str(form, "method", "aggressive"),
            mask_channel=_form_str(form, "mask_channel", "grayscale"),
            threshold=_form_float(form, "threshold", 0.5),
            dilate_radius=_form_int(form, "dilate_radius", 1),
            feather_radius=_form_int(form, "feather_radius", 1),
            strength=_form_float(form, "strength", 1.0),
            min_component_area=_form_int(form, "min_component_area", 1),
            max_component_area=_form_optional_int(form, "max_component_area", 200000),
            padding=_form_int(form, "padding", 32),
            debug_dir=debug_dir,
        )
        if cfg.method not in REPAIR_METHODS:
            raise ValueError(f"Unsupported method: {cfg.method}")
        if cfg.mask_channel not in MASK_CHANNELS:
            raise ValueError(f"Unsupported mask channel: {cfg.mask_channel}")

        image = read_image(image_path)
        mask = read_image(mask_path)
        result = repair_image(image.pixels, mask.pixels, cfg)
        output_path = run_dir / "repaired.png"
        write_image(output_path, result.repaired_image)

        return {
            "ok": True,
            "run_id": run_id,
            "input_url": _output_url(image_path),
            "mask_url": _output_url(debug_dir / "binary_mask.png"),
            "soft_mask_url": _output_url(debug_dir / "soft_mask.png"),
            "repaired_url": _output_url(output_path),
            "diff_url": _output_url(debug_dir / "diff_visualization.png"),
            "metrics_url": _output_url(debug_dir / "metrics.json"),
            "metrics": result.metrics,
        }

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Dust Mask Repair local test UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), RepairWebHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Dust Mask Repair UI: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _required_file(form: cgi.FieldStorage, name: str) -> cgi.FieldStorage:
    field = form[name] if name in form else None
    if field is None or not getattr(field, "filename", None):
        raise ValueError(f"Missing uploaded file: {name}")
    return field


def _save_upload(field: cgi.FieldStorage, directory: Path, stem: str, default_suffix: str) -> Path:
    suffix = Path(field.filename or "").suffix.lower() or default_suffix
    if suffix not in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        raise ValueError(f"Unsupported uploaded file extension: {suffix}")
    path = directory / f"{stem}{suffix}"
    with path.open("wb") as output:
        field.file.seek(0)
        shutil.copyfileobj(field.file, output)
    return path


def _form_str(form: cgi.FieldStorage, name: str, default: str) -> str:
    return str(form.getfirst(name, default))


def _form_int(form: cgi.FieldStorage, name: str, default: int) -> int:
    return int(str(form.getfirst(name, default)))


def _form_optional_int(form: cgi.FieldStorage, name: str, default: int | None) -> int | None:
    value = str(form.getfirst(name, "" if default is None else default)).strip()
    if value == "":
        return None
    return int(value)


def _form_float(form: cgi.FieldStorage, name: str, default: float) -> float:
    return float(str(form.getfirst(name, default)))


def _output_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix()
    return f"/outputs/{rel}"


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix in (".tif", ".tiff"):
        return "image/tiff"
    return "application/octet-stream"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
