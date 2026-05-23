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
from .red_highlight import RedHighlightConfig, detect_red_highlight_source_image
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
        if parsed.path not in ("/api/repair", "/api/repair-red"):
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            if parsed.path == "/api/repair-red":
                response = self._handle_repair_red()
            else:
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

        cfg = _repair_config_from_form(form, debug_dir)

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

    def _handle_repair_red(self) -> dict[str, object]:
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
        red_field = _required_file(form, "red_image")
        run_id = uuid.uuid4().hex[:12]
        run_dir = OUTPUT_ROOT / run_id
        debug_dir = run_dir / "debug"
        red_debug_dir = run_dir / "red_highlight"
        run_dir.mkdir(parents=True, exist_ok=True)

        image_path = _save_upload(image_field, run_dir, "input", ".png")
        red_path = _save_upload(red_field, run_dir, "red_input", ".png")

        repair_cfg = _repair_config_from_form(form, debug_dir)
        red_cfg = RedHighlightConfig(
            detection_long_edge=_form_int(form, "detection_long_edge", 1280),
            mask_edge_mode=_form_str(form, "mask_edge_mode", "normal"),
            full_resolution_refine=_form_bool(form, "full_resolution_refine", True),
            red_min=_form_optional_float(form, "red_min"),
            red_excess_min=_form_optional_float(form, "red_excess_min"),
            red_ratio_min=_form_optional_float(form, "red_ratio_min"),
            contrast_red_min=_form_optional_float(form, "contrast_red_min"),
            contrast_excess_min=_form_optional_float(form, "contrast_excess_min"),
            glow_signal_min=_form_optional_float(form, "glow_signal_min"),
            contrast_glow_min=_form_optional_float(form, "contrast_glow_min"),
            hot_core_value_min=_form_optional_float(form, "hot_core_value_min"),
            hot_core_contrast_min=_form_optional_float(form, "hot_core_contrast_min"),
            threshold_sensitivity=_form_float(form, "threshold_sensitivity", 1.0),
            min_area=_form_int(form, "min_red_area", 1),
            max_area=_form_int(form, "max_red_area", 1400),
            max_dim=_form_int(form, "max_red_dim", 95),
            include_long_scratches=_form_bool(form, "include_long_scratches", False),
            min_scratch_aspect=_form_float(form, "min_scratch_aspect", 5.0),
            max_scratch_area=_form_int(form, "max_scratch_area", 9000),
            max_scratch_dim=_form_int(form, "max_scratch_dim", 720),
            max_scratch_width=_form_int(form, "max_scratch_width", 48),
            suppress_border_glow=_form_bool(form, "suppress_border_glow", True),
        )

        image = read_image(image_path)
        red_image = read_image(red_path)
        if image.pixels.shape[:2] != red_image.pixels.shape[:2]:
            raise ValueError(
                "image and red-image dimensions differ: "
                f"image={image.pixels.shape[1]}x{image.pixels.shape[0]}, "
                f"red_image={red_image.pixels.shape[1]}x{red_image.pixels.shape[0]}"
            )

        red_result = detect_red_highlight_source_image(red_image.pixels, red_cfg)
        red_debug_dir.mkdir(parents=True, exist_ok=True)
        red_mask_path = red_debug_dir / "red_highlight_mask.png"
        red_preview_mask_path = red_debug_dir / "preview_mask.png"
        red_overlay_path = red_debug_dir / "overlay.png"
        red_overlay_preview_path = red_debug_dir / "overlay_preview.png"
        red_manifest_path = red_debug_dir / "manifest.json"
        write_image(red_mask_path, red_result.mask)
        write_image(red_preview_mask_path, red_result.preview_mask)
        write_image(red_overlay_path, red_result.overlay)
        write_image(red_overlay_preview_path, red_result.overlay_preview)
        red_manifest_path.write_text(json.dumps(red_result.manifest, indent=2, sort_keys=True), encoding="utf-8")

        result = repair_image(image.pixels, red_result.mask, repair_cfg)
        output_path = run_dir / "repaired.png"
        write_image(output_path, result.repaired_image)

        return {
            "ok": True,
            "run_id": run_id,
            "input_url": _output_url(image_path),
            "red_input_url": _output_url(red_path),
            "generated_mask_url": _output_url(red_mask_path),
            "red_preview_mask_url": _output_url(red_preview_mask_path),
            "red_overlay_url": _output_url(red_overlay_path),
            "red_overlay_preview_url": _output_url(red_overlay_preview_path),
            "red_manifest_url": _output_url(red_manifest_path),
            "mask_url": _output_url(debug_dir / "binary_mask.png"),
            "soft_mask_url": _output_url(debug_dir / "soft_mask.png"),
            "repaired_url": _output_url(output_path),
            "diff_url": _output_url(debug_dir / "diff_visualization.png"),
            "metrics_url": _output_url(debug_dir / "metrics.json"),
            "metrics": result.metrics,
            "red_highlight": red_result.manifest,
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


def _form_optional_float(form: cgi.FieldStorage, name: str) -> float | None:
    value = str(form.getfirst(name, "")).strip()
    if value == "":
        return None
    return float(value)


def _form_bool(form: cgi.FieldStorage, name: str, default: bool) -> bool:
    value = str(form.getfirst(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _repair_config_from_form(form: cgi.FieldStorage, debug_dir: Path) -> RepairConfig:
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
    return cfg


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
