from __future__ import annotations

import base64
import io
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from textwrap import wrap
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image


XMP_DMR_NS = "https://dust-mask-repair.local/ns/1.0/"
XMP_FORMAT = "dust-mask-repair-xmp-v1"
SILVERTURN_XMP_NS = "https://silverturn.local/ns/dust-mask/1.0/"
SILVERTURN_XMP_FORMAT = "silverturn-dust-mask-xmp-v1"
MASK_OUTPUT_MODE_LEGACY_PLUS_XMP = "legacy_plus_xmp"


@dataclass(frozen=True)
class MaskXmpData:
    mask: np.ndarray
    manifest: dict[str, Any]
    mask_mapping: dict[str, Any]
    format: str
    role: str
    namespace: str
    source_file_name: str
    target_file_name: str
    output_mode: str
    adobe_native_mask: bool = False


def write_mask_xmp(
    path: str | Path,
    *,
    mask: np.ndarray,
    manifest: dict[str, Any],
    source_path: str | Path | None = None,
    target_path: str | Path | None = None,
    mask_mapping: dict[str, Any] | None = None,
    role: str = "white_dust_detection",
) -> dict[str, Any]:
    """Write a host-neutral XMP sidecar that carries the detected mask.

    This is intentionally a custom XMP namespace. It does not claim Adobe
    Camera Raw / Lightroom local-mask compatibility.
    """

    xmp_path = Path(path)
    xmp_path.parent.mkdir(parents=True, exist_ok=True)

    mask_u8 = _binary_mask_u8(mask)
    height, width = mask_u8.shape
    mask_pixels = int(np.count_nonzero(mask_u8))
    encoded_png = _encode_png_base64(mask_u8)
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=str)
    mapping_json = json.dumps(mask_mapping or {}, ensure_ascii=False, sort_keys=True, default=str)
    created_utc = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    source_name = Path(source_path).name if source_path is not None else ""
    target_name = Path(target_path).name if target_path is not None else ""

    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="dust-mask-repair">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dmr="{XMP_DMR_NS}"
      dmr:format="{XMP_FORMAT}"
      dmr:role="{_xml_attr(role)}"
      dmr:createdUtc="{_xml_attr(created_utc)}"
      dmr:maskWidth="{width}"
      dmr:maskHeight="{height}"
      dmr:maskPixels="{mask_pixels}"
      dmr:outputMode="{MASK_OUTPUT_MODE_LEGACY_PLUS_XMP}"
      dmr:sourceFileName="{_xml_attr(source_name)}"
      dmr:targetFileName="{_xml_attr(target_name)}">
      <dmr:MaskPngEncoding>base64/png</dmr:MaskPngEncoding>
      <dmr:MaskPng>
{_indent_text(encoded_png, 8)}
      </dmr:MaskPng>
      <dmr:ManifestJson>{_xml_text(manifest_json)}</dmr:ManifestJson>
      <dmr:MaskMappingJson>{_xml_text(mapping_json)}</dmr:MaskMappingJson>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""
    xmp_path.write_text(body, encoding="utf-8")
    return {
        "path": str(xmp_path),
        "format": XMP_FORMAT,
        "role": role,
        "mask_width": width,
        "mask_height": height,
        "mask_pixels": mask_pixels,
        "output_mode": MASK_OUTPUT_MODE_LEGACY_PLUS_XMP,
        "mask_png_encoding": "base64/png",
        "custom_namespace": XMP_DMR_NS,
        "adobe_native_mask": False,
    }


def read_mask_xmp(path: str | Path) -> MaskXmpData:
    """Read a custom Dust Mask Repair / SilverTurn mask XMP sidecar."""

    xmp_path = Path(path)
    root = ET.parse(xmp_path).getroot()
    description, namespace, xmp_format = _find_supported_description(root)
    role = description.attrib.get(f"{{{namespace}}}role", "")
    output_mode = description.attrib.get(f"{{{namespace}}}outputMode", "")
    source_name = description.attrib.get(f"{{{namespace}}}sourceFileName", "")
    target_name = description.attrib.get(f"{{{namespace}}}targetFileName", "")
    if namespace == SILVERTURN_XMP_NS:
        source_name = description.attrib.get(f"{{{namespace}}}sourceName", source_name)

    encoding = _find_required_text(root, namespace, "MaskPngEncoding").strip()
    if encoding != "base64/png":
        raise ValueError(f"Unsupported XMP mask encoding: {encoding}")

    encoded_png = _find_required_text(root, namespace, "MaskPng")
    png_bytes = base64.b64decode("".join(encoded_png.split()), validate=True)
    with Image.open(io.BytesIO(png_bytes)) as image:
        mask = np.asarray(image.convert("L"))
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    manifest_text = _find_optional_text(root, namespace, "ManifestJson")
    if manifest_text is None:
        manifest_text = _find_optional_text(root, namespace, "ReportJson")
    manifest = json.loads(manifest_text) if manifest_text else {}

    mapping_text = _find_optional_text(root, namespace, "MaskMappingJson")
    mask_mapping = json.loads(mapping_text) if mapping_text else {}

    return MaskXmpData(
        mask=mask,
        manifest=manifest,
        mask_mapping=mask_mapping,
        format=xmp_format,
        role=role,
        namespace=namespace,
        source_file_name=source_name,
        target_file_name=target_name,
        output_mode=output_mode,
        adobe_native_mask=False,
    )


def _binary_mask_u8(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"XMP mask output requires a 2D mask, got {arr.shape}")
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _find_supported_description(root: ET.Element) -> tuple[ET.Element, str, str]:
    for namespace, expected_format in (
        (XMP_DMR_NS, XMP_FORMAT),
        (SILVERTURN_XMP_NS, SILVERTURN_XMP_FORMAT),
    ):
        format_attr = f"{{{namespace}}}format"
        for description in root.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"):
            xmp_format = description.attrib.get(format_attr)
            if xmp_format == expected_format:
                return description, namespace, xmp_format
    raise ValueError("Unsupported mask XMP format")


def _find_required_text(root: ET.Element, namespace: str, tag: str) -> str:
    value = _find_optional_text(root, namespace, tag)
    if value is None:
        raise ValueError(f"XMP mask is missing {tag}")
    return value


def _find_optional_text(root: ET.Element, namespace: str, tag: str) -> str | None:
    return root.findtext(f".//{{{namespace}}}{tag}")


def _encode_png_base64(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "\n".join(wrap(encoded, 76))


def _indent_text(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def _xml_text(value: str) -> str:
    return escape(value)


def _xml_attr(value: str) -> str:
    return escape(value, {'"': "&quot;", "'": "&apos;"})
