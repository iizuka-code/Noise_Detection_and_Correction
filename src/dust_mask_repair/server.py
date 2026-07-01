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

import numpy as np
from PIL import Image

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import cgi

from .config import MASK_CHANNELS, REPAIR_METHODS, RepairConfig
from .io import RAW_EXTENSIONS, read_image, write_image
from .red_highlight import RedHighlightConfig
from .repair import repair_image
from .white_dust import (
    WhiteDustConfig,
    detect_white_dust_proxy_image,
    detect_white_dust_source_image,
    make_white_dust_input_preview,
)
from .workflow import repair_image_from_red_highlight
from .xmp import MASK_OUTPUT_MODE_LEGACY_PLUS_XMP, write_mask_xmp


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
        if parsed.path not in ("/api/repair", "/api/repair-red", "/api/detect-white-dust", "/api/repair-white-dust"):
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            if parsed.path == "/api/repair-white-dust":
                response = self._handle_repair_white_dust()
            elif parsed.path == "/api/detect-white-dust":
                response = self._handle_detect_white_dust()
            elif parsed.path == "/api/repair-red":
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
        mask_path = _save_upload(mask_field, run_dir, "mask", ".png", allow_raw=False)

        cfg = _repair_config_from_form(form, debug_dir)

        image = read_image(image_path)
        mask = read_image(mask_path)
        input_preview_path = run_dir / "input_preview.png"
        write_image(input_preview_path, make_white_dust_input_preview(image.pixels, 1600))
        result = repair_image(image.pixels, mask.pixels, cfg)
        output_path = run_dir / "repaired.png"
        write_image(output_path, result.repaired_image)

        return {
            "ok": True,
            "run_id": run_id,
            "input_url": _output_url(image_path),
            "input_preview_url": _output_url(input_preview_path),
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
            visual_artifacts=True,
        )

        image = read_image(image_path)
        red_image = read_image(red_path)
        input_preview_path = run_dir / "input_preview.png"
        write_image(input_preview_path, make_white_dust_input_preview(image.pixels, 1600))
        workflow_result = repair_image_from_red_highlight(
            image.pixels,
            red_image.pixels,
            red_config=red_cfg,
            repair_config=repair_cfg,
        )

        red_result = workflow_result.red_highlight
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

        result = workflow_result.repair
        output_path = run_dir / "repaired.png"
        write_image(output_path, result.repaired_image)

        return {
            "ok": True,
            "run_id": run_id,
            "input_url": _output_url(image_path),
            "input_preview_url": _output_url(input_preview_path),
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

    def _handle_detect_white_dust(self) -> dict[str, object]:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        source_field = _required_file(form, "source")
        run_id = uuid.uuid4().hex[:12]
        run_dir = OUTPUT_ROOT / run_id
        dust_dir = run_dir / "white_dust"
        run_dir.mkdir(parents=True, exist_ok=True)

        source_path = _save_upload(source_field, run_dir, "source", ".png")
        cfg = WhiteDustConfig(
            detection_long_edge=_form_int(form, "detection_long_edge", 1024),
            local_radius=_form_int(form, "local_radius", 0),
            background_mode=_form_str(form, "background_mode", "dark"),
            mask_edge_mode=_form_str(form, "mask_edge_mode", "normal"),
            threshold_sensitivity=_form_float(form, "threshold_sensitivity", 1.0),
            value_min=_form_optional_float(form, "value_min"),
            luma_min=_form_optional_float(form, "luma_min"),
            white_floor_min=_form_optional_float(form, "white_floor_min"),
            value_contrast_min=_form_optional_float(form, "value_contrast_min"),
            bright_contrast_min=_form_optional_float(form, "bright_contrast_min"),
            white_contrast_min=_form_optional_float(form, "white_contrast_min"),
            whiteness_min=_form_float(form, "whiteness_min", 0.60),
            min_area=_form_int(form, "min_area", 2),
            max_area=_form_int(form, "max_area", 12000),
            max_dim=_form_int(form, "max_dim", 900),
            max_thickness=_form_float(form, "max_thickness", 54.0),
            require_brown_background=_form_bool(form, "require_brown_background", True),
            brown_blue_deficit_min=_form_float(form, "brown_blue_deficit_min", 8.0),
            brown_red_blue_ratio_min=_form_float(form, "brown_red_blue_ratio_min", 1.10),
            brown_luma_max=_form_optional_float(form, "brown_luma_max", 170.0),
            dark_luma_max=_form_float(form, "dark_luma_max", 72.0),
            dark_value_max=_form_float(form, "dark_value_max", 96.0),
            focus_margin_x=_form_float(form, "focus_margin_x", 0.0),
            focus_margin_y=_form_float(form, "focus_margin_y", 0.0),
            visual_artifacts=True,
        )
        raw_source = source_path.suffix.lower() in RAW_EXTENSIONS
        image = read_image(source_path, raw_half_size=raw_source, raw_output_bps=8 if raw_source else 16)
        if raw_source:
            result = detect_white_dust_proxy_image(image.pixels, cfg)
        else:
            result = detect_white_dust_source_image(image.pixels, cfg)
        manifest = dict(result.manifest)
        manifest["source_mode"] = image.color_mode
        manifest["source_metadata"] = image.metadata
        manifest["raw_fast_proxy"] = raw_source

        dust_dir.mkdir(parents=True, exist_ok=True)
        source_preview_path = dust_dir / "source_preview.png"
        mask_path = dust_dir / "white_dust_mask.png"
        preview_mask_path = dust_dir / "preview_mask.png"
        overlay_path = dust_dir / "overlay.png"
        overlay_preview_path = dust_dir / "overlay_preview.png"
        score_path = dust_dir / "white_dust_score.png"
        manifest_path = dust_dir / "manifest.json"
        xmp_path = dust_dir / "white_dust_mask.xmp"
        write_image(source_preview_path, make_white_dust_input_preview(image.pixels, cfg.detection_long_edge))
        write_image(mask_path, result.mask)
        write_image(preview_mask_path, result.preview_mask)
        write_image(overlay_preview_path, result.overlay_preview)
        if raw_source:
            shutil.copyfile(overlay_preview_path, overlay_path)
        else:
            write_image(overlay_path, result.overlay)
        write_image(score_path, result.score_map)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        xmp_summary = write_mask_xmp(
            xmp_path,
            mask=result.mask,
            manifest=manifest,
            source_path=source_path,
            role="white_dust_detection",
        )

        return {
            "ok": True,
            "run_id": run_id,
            "source_url": _output_url(source_path),
            "source_preview_url": _output_url(source_preview_path),
            "generated_mask_url": _output_url(mask_path),
            "preview_mask_url": _output_url(preview_mask_path),
            "overlay_url": _output_url(overlay_path),
            "overlay_preview_url": _output_url(overlay_preview_path),
            "score_url": _output_url(score_path),
            "manifest_url": _output_url(manifest_path),
            "xmp_url": _output_url(xmp_path),
            "xmp": xmp_summary,
            "mask_output_mode": MASK_OUTPUT_MODE_LEGACY_PLUS_XMP,
            "white_dust": manifest,
        }

    def _handle_repair_white_dust(self) -> dict[str, object]:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        source_field = _required_file(form, "source")
        image_field = _required_file(form, "image")
        run_id = uuid.uuid4().hex[:12]
        run_dir = OUTPUT_ROOT / run_id
        dust_dir = run_dir / "white_dust"
        debug_dir = run_dir / "debug"
        run_dir.mkdir(parents=True, exist_ok=True)

        source_path = _save_upload(source_field, run_dir, "source", ".png")
        image_path = _save_upload(image_field, run_dir, "input", ".png", allow_raw=True)

        cfg = _white_dust_config_from_form(form)
        repair_cfg = _repair_config_from_form(
            form,
            debug_dir,
            default_method="kl",
            default_dilate_radius=2,
            default_feather_radius=1,
        )

        raw_source = source_path.suffix.lower() in RAW_EXTENSIONS
        source_image = read_image(source_path, raw_half_size=raw_source, raw_output_bps=8 if raw_source else 16)
        if raw_source:
            dust_result = detect_white_dust_proxy_image(source_image.pixels, cfg)
        else:
            dust_result = detect_white_dust_source_image(source_image.pixels, cfg)
        dust_manifest = dict(dust_result.manifest)
        dust_manifest["source_mode"] = source_image.color_mode
        dust_manifest["source_metadata"] = source_image.metadata
        dust_manifest["raw_fast_proxy"] = raw_source

        target_image = read_image(image_path)
        fitted_mask, mask_mapping = _fit_mask_to_image(dust_result.mask, target_image.pixels.shape[:2])
        repair_result = repair_image(target_image.pixels, fitted_mask, repair_cfg)

        dust_dir.mkdir(parents=True, exist_ok=True)
        source_preview_path = dust_dir / "source_preview.png"
        detected_mask_path = dust_dir / "white_dust_mask.png"
        preview_mask_path = dust_dir / "preview_mask.png"
        fitted_mask_path = dust_dir / "fitted_repair_mask.png"
        target_mask_overlay_path = dust_dir / "target_mask_overlay.png"
        overlay_path = dust_dir / "overlay.png"
        overlay_preview_path = dust_dir / "overlay_preview.png"
        score_path = dust_dir / "white_dust_score.png"
        manifest_path = dust_dir / "manifest.json"
        detected_xmp_path = dust_dir / "white_dust_mask.xmp"
        fitted_xmp_path = dust_dir / "fitted_repair_mask.xmp"
        input_preview_path = run_dir / "input_preview.png"
        repaired_path = _repair_output_path(run_dir, image_path)

        write_image(source_preview_path, make_white_dust_input_preview(source_image.pixels, cfg.detection_long_edge))
        write_image(detected_mask_path, dust_result.mask)
        write_image(preview_mask_path, dust_result.preview_mask)
        write_image(fitted_mask_path, fitted_mask)
        write_image(target_mask_overlay_path, _make_target_mask_overlay(target_image.pixels, fitted_mask))
        write_image(overlay_preview_path, dust_result.overlay_preview)
        if raw_source:
            shutil.copyfile(overlay_preview_path, overlay_path)
        else:
            write_image(overlay_path, dust_result.overlay)
        write_image(score_path, dust_result.score_map)
        write_image(input_preview_path, make_white_dust_input_preview(target_image.pixels, 1600))
        write_image(repaired_path, repair_result.repaired_image)

        dust_manifest["mask_mapping"] = mask_mapping
        manifest_path.write_text(json.dumps(dust_manifest, indent=2, sort_keys=True), encoding="utf-8")
        xmp_summary = write_mask_xmp(
            detected_xmp_path,
            mask=dust_result.mask,
            manifest=dust_manifest,
            source_path=source_path,
            role="white_dust_detection",
        )
        fitted_xmp_summary = write_mask_xmp(
            fitted_xmp_path,
            mask=fitted_mask,
            manifest=dust_manifest,
            source_path=source_path,
            target_path=image_path,
            mask_mapping=mask_mapping,
            role="white_dust_fitted_repair",
        )

        return {
            "ok": True,
            "run_id": run_id,
            "source_url": _output_url(source_path),
            "source_preview_url": _output_url(source_preview_path),
            "input_url": _output_url(input_preview_path if image_path.suffix.lower() in RAW_EXTENSIONS else image_path),
            "input_original_url": _output_url(image_path),
            "input_preview_url": _output_url(input_preview_path),
            "repaired_url": _output_url(repaired_path),
            "generated_mask_url": _output_url(detected_mask_path),
            "fitted_mask_url": _output_url(fitted_mask_path),
            "target_mask_overlay_url": _output_url(target_mask_overlay_path),
            "preview_mask_url": _output_url(preview_mask_path),
            "overlay_url": _output_url(overlay_path),
            "overlay_preview_url": _output_url(overlay_preview_path),
            "score_url": _output_url(score_path),
            "manifest_url": _output_url(manifest_path),
            "xmp_url": _output_url(detected_xmp_path),
            "fitted_xmp_url": _output_url(fitted_xmp_path),
            "mask_url": _output_url(debug_dir / "binary_mask.png"),
            "soft_mask_url": _output_url(debug_dir / "soft_mask.png"),
            "diff_url": _output_url(debug_dir / "diff_visualization.png"),
            "metrics_url": _output_url(debug_dir / "metrics.json"),
            "metrics": repair_result.metrics,
            "white_dust": dust_manifest,
            "mask_mapping": mask_mapping,
            "xmp": xmp_summary,
            "fitted_xmp": fitted_xmp_summary,
            "mask_output_mode": MASK_OUTPUT_MODE_LEGACY_PLUS_XMP,
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


def _save_upload(
    field: cgi.FieldStorage,
    directory: Path,
    stem: str,
    default_suffix: str,
    *,
    allow_raw: bool = True,
) -> Path:
    suffix = Path(field.filename or "").suffix.lower() or default_suffix
    allowed = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    if allow_raw:
        allowed |= RAW_EXTENSIONS
    if suffix not in allowed:
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


def _form_optional_float(form: cgi.FieldStorage, name: str, default: float | None = None) -> float | None:
    value = str(form.getfirst(name, "" if default is None else default)).strip()
    if value == "":
        return None
    return float(value)


def _form_bool(form: cgi.FieldStorage, name: str, default: bool) -> bool:
    value = str(form.getfirst(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _white_dust_config_from_form(form: cgi.FieldStorage) -> WhiteDustConfig:
    return WhiteDustConfig(
        detection_long_edge=_form_int(form, "detection_long_edge", 1024),
        local_radius=_form_int(form, "local_radius", 0),
        background_mode=_form_str(form, "background_mode", "dark"),
        mask_edge_mode=_form_str(form, "mask_edge_mode", "normal"),
        threshold_sensitivity=_form_float(form, "threshold_sensitivity", 1.0),
        value_min=_form_optional_float(form, "value_min"),
        luma_min=_form_optional_float(form, "luma_min"),
        white_floor_min=_form_optional_float(form, "white_floor_min"),
        value_contrast_min=_form_optional_float(form, "value_contrast_min"),
        bright_contrast_min=_form_optional_float(form, "bright_contrast_min"),
        white_contrast_min=_form_optional_float(form, "white_contrast_min"),
        whiteness_min=_form_float(form, "whiteness_min", 0.60),
        min_area=_form_int(form, "min_area", 2),
        max_area=_form_int(form, "max_area", 12000),
        max_dim=_form_int(form, "max_dim", 900),
        max_thickness=_form_float(form, "max_thickness", 54.0),
        require_brown_background=_form_bool(form, "require_brown_background", True),
        brown_blue_deficit_min=_form_float(form, "brown_blue_deficit_min", 8.0),
        brown_red_blue_ratio_min=_form_float(form, "brown_red_blue_ratio_min", 1.10),
        brown_luma_max=_form_optional_float(form, "brown_luma_max", 170.0),
        dark_luma_max=_form_float(form, "dark_luma_max", 72.0),
        dark_value_max=_form_float(form, "dark_value_max", 96.0),
        focus_margin_x=_form_float(form, "focus_margin_x", 0.0),
        focus_margin_y=_form_float(form, "focus_margin_y", 0.0),
        visual_artifacts=True,
    )


def _repair_config_from_form(
    form: cgi.FieldStorage,
    debug_dir: Path,
    *,
    default_method: str = "aggressive",
    default_dilate_radius: int = 1,
    default_feather_radius: int = 1,
) -> RepairConfig:
    cfg = RepairConfig(
        method=_form_str(form, "method", default_method),
        mask_channel=_form_str(form, "mask_channel", "grayscale"),
        threshold=_form_float(form, "threshold", 0.5),
        dilate_radius=_form_int(form, "dilate_radius", default_dilate_radius),
        feather_radius=_form_int(form, "feather_radius", default_feather_radius),
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


def _fit_mask_to_image(mask: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, object]]:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = np.max(mask_arr[:, :, :3], axis=2)
    if mask_arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape for fitting: {mask_arr.shape}")
    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Unsupported target shape for mask fitting: {target_shape}")

    source_height, source_width = int(mask_arr.shape[0]), int(mask_arr.shape[1])
    source_aspect = source_width / float(source_height)
    target_aspect = target_width / float(target_height)
    x0, y0, x1, y1 = 0, 0, source_width, source_height
    mode = "resize"
    if abs(source_aspect - target_aspect) / max(source_aspect, target_aspect) > 0.005:
        mode = "center_crop_resize"
        if source_aspect > target_aspect:
            crop_width = max(1, min(source_width, int(round(source_height * target_aspect))))
            x0 = max(0, (source_width - crop_width) // 2)
            x1 = x0 + crop_width
        else:
            crop_height = max(1, min(source_height, int(round(source_width / target_aspect))))
            y0 = max(0, (source_height - crop_height) // 2)
            y1 = y0 + crop_height

    cropped = mask_arr[y0:y1, x0:x1]
    resized = np.asarray(
        Image.fromarray(cropped.astype(np.uint8)).resize(
            (target_width, target_height),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    fitted = np.where(resized > 0, 255, 0).astype(np.uint8)
    mapping: dict[str, object] = {
        "mode": mode,
        "source_shape": [source_height, source_width],
        "target_shape": [target_height, target_width],
        "source_aspect": round(source_aspect, 8),
        "target_aspect": round(target_aspect, 8),
        "crop_box_xyxy": [x0, y0, x1, y1],
        "fitted_mask_pixels": int(np.count_nonzero(fitted)),
    }
    return fitted, mapping


def _make_target_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[2] < 3:
        raise ValueError(f"Unsupported image shape for mask overlay: {image_arr.shape}")
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = np.max(mask_arr[:, :, :3], axis=2)
    if mask_arr.shape != image_arr.shape[:2]:
        raise ValueError(
            "image and mask dimensions differ for overlay: "
            f"image={image_arr.shape[1]}x{image_arr.shape[0]}, "
            f"mask={mask_arr.shape[1]}x{mask_arr.shape[0]}"
        )

    rgb = image_arr[:, :, :3]
    if rgb.dtype == np.uint8:
        base = rgb.copy()
    elif np.issubdtype(rgb.dtype, np.integer):
        max_value = float(np.iinfo(rgb.dtype).max)
        base = np.rint(np.clip(rgb.astype(np.float32) / max_value, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        base = np.rint(np.clip(rgb.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)

    active = mask_arr > 0
    overlay = base.copy()
    if active.any():
        color = np.array([255.0, 32.0, 16.0], dtype=np.float32)
        alpha = 0.68
        blended = base[active].astype(np.float32) * (1.0 - alpha) + color * alpha
        overlay[active] = np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)
    return overlay


def _repair_output_path(run_dir: Path, image_path: Path) -> Path:
    suffix = image_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return run_dir / "repaired.jpg"
    if suffix in (".tif", ".tiff"):
        return run_dir / "repaired.tif"
    return run_dir / "repaired.png"


def _output_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix()
    return f"/outputs/{rel}"


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    if suffix == ".xmp":
        return "application/rdf+xml; charset=utf-8"
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
