from __future__ import annotations

import base64
import io
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from dust_mask_repair.xmp import (
    MASK_OUTPUT_MODE_LEGACY_PLUS_XMP,
    SILVERTURN_XMP_FORMAT,
    SILVERTURN_XMP_NS,
    XMP_DMR_NS,
    XMP_FORMAT,
    read_mask_xmp,
    write_mask_xmp,
)


def test_write_mask_xmp_embeds_binary_png_and_manifest(tmp_path) -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    manifest = {
        "detector_version": "dust_on_dark_or_brown_v2",
        "final_mask_pixels": int(np.count_nonzero(mask)),
        "note": "source & target metadata must be XML-safe",
    }
    xmp_path = tmp_path / "mask.xmp"

    summary = write_mask_xmp(
        xmp_path,
        mask=mask,
        manifest=manifest,
        source_path="inspection & source.arw",
        target_path="repair target.jpg",
        mask_mapping={"mode": "center_crop_resize", "target_shape": [8, 10]},
        role="white_dust_fitted_repair",
    )

    assert summary["format"] == XMP_FORMAT
    assert summary["mask_width"] == 10
    assert summary["mask_height"] == 8
    assert summary["mask_pixels"] == 12
    assert summary["output_mode"] == MASK_OUTPUT_MODE_LEGACY_PLUS_XMP
    assert summary["adobe_native_mask"] is False

    root = ET.parse(xmp_path).getroot()
    namespaces = {"dmr": XMP_DMR_NS}
    description = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert description is not None
    assert description.attrib[f"{{{XMP_DMR_NS}}}format"] == XMP_FORMAT
    assert description.attrib[f"{{{XMP_DMR_NS}}}role"] == "white_dust_fitted_repair"
    assert description.attrib[f"{{{XMP_DMR_NS}}}sourceFileName"] == "inspection & source.arw"

    encoded = root.findtext(".//dmr:MaskPng", namespaces=namespaces)
    assert encoded is not None
    png_bytes = base64.b64decode("".join(encoded.split()))
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    manifest_json = root.findtext(".//dmr:ManifestJson", namespaces=namespaces)
    assert manifest_json is not None
    assert json.loads(manifest_json)["final_mask_pixels"] == 12

    mapping_json = root.findtext(".//dmr:MaskMappingJson", namespaces=namespaces)
    assert mapping_json is not None
    assert json.loads(mapping_json)["mode"] == "center_crop_resize"


def test_read_mask_xmp_round_trips_dust_mask_repair_sidecar(tmp_path) -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    xmp_path = tmp_path / "mask.xmp"

    write_mask_xmp(
        xmp_path,
        mask=mask,
        manifest={"final_mask_pixels": 12},
        source_path="source.arw",
        target_path="target.jpg",
        mask_mapping={"mode": "resize"},
        role="white_dust_fitted_repair",
    )

    loaded = read_mask_xmp(xmp_path)

    assert loaded.format == XMP_FORMAT
    assert loaded.namespace == XMP_DMR_NS
    assert loaded.role == "white_dust_fitted_repair"
    assert loaded.source_file_name == "source.arw"
    assert loaded.target_file_name == "target.jpg"
    assert loaded.output_mode == MASK_OUTPUT_MODE_LEGACY_PLUS_XMP
    assert loaded.manifest["final_mask_pixels"] == 12
    assert loaded.mask_mapping["mode"] == "resize"
    assert np.array_equal(loaded.mask, mask)


def test_read_mask_xmp_accepts_silverturn_cli_sidecar(tmp_path) -> None:
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    xmp_path = tmp_path / "silverturn_mask.xmp"
    xmp_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="silverturn">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dmr="{SILVERTURN_XMP_NS}"
      dmr:format="{SILVERTURN_XMP_FORMAT}"
      dmr:role="silverturn_dust_detection"
      dmr:outputMode="{MASK_OUTPUT_MODE_LEGACY_PLUS_XMP}"
      dmr:sourceName="frame01"
      dmr:maskWidth="5"
      dmr:maskHeight="4">
      <dmr:MaskPngEncoding>base64/png</dmr:MaskPngEncoding>
      <dmr:MaskPng>{encoded}</dmr:MaskPng>
      <dmr:ReportJson>{{"masked_pixels":4}}</dmr:ReportJson>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
        encoding="utf-8",
    )

    loaded = read_mask_xmp(xmp_path)

    assert loaded.format == SILVERTURN_XMP_FORMAT
    assert loaded.namespace == SILVERTURN_XMP_NS
    assert loaded.role == "silverturn_dust_detection"
    assert loaded.output_mode == MASK_OUTPUT_MODE_LEGACY_PLUS_XMP
    assert loaded.source_file_name == "frame01"
    assert loaded.manifest["masked_pixels"] == 4
    assert np.array_equal(loaded.mask, mask)
