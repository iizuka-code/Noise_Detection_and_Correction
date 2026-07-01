from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dust_mask_repair.io import read_image
from dust_mask_repair.white_dust import WhiteDustConfig, run_white_dust_detector


ROOT = Path(__file__).resolve().parents[1]


class _FakeRawFile:
    def __init__(self, calls: dict[str, object], pixels: np.ndarray) -> None:
        self._calls = calls
        self._pixels = pixels

    def __enter__(self) -> "_FakeRawFile":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        return None

    def postprocess(self, **kwargs: object) -> np.ndarray:
        self._calls["postprocess"] = kwargs
        return self._pixels


def _install_fake_rawpy(monkeypatch: pytest.MonkeyPatch, pixels: np.ndarray) -> dict[str, object]:
    calls: dict[str, object] = {}

    def imread(path: str) -> _FakeRawFile:
        calls["path"] = path
        return _FakeRawFile(calls, pixels)

    fake_rawpy = SimpleNamespace(ColorSpace=SimpleNamespace(sRGB="srgb"), imread=imread)
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)
    return calls


def _dark_raw_fixture() -> np.ndarray:
    image = np.zeros((80, 96, 3), dtype=np.uint16)
    image[:, :] = [4200, 4300, 5000]
    image[38:43, 45:50] = [61000, 59500, 57000]
    image[56:60, 65:69] = [56000, 26000, 42000]
    return image


def test_raw_input_requires_optional_rawpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rawpy", None)

    with pytest.raises(ValueError, match="RAW input requires optional dependency: rawpy"):
        read_image("missing.dng")


def test_read_image_decodes_rw2_with_rawpy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_rawpy(monkeypatch, _dark_raw_fixture())

    image = read_image("source.rw2")

    assert image.pixels.dtype == np.uint16
    assert image.pixels.shape == (80, 96, 3)
    assert image.metadata["format"] == "RAW"
    assert image.metadata["reader"] == "rawpy"
    assert image.metadata["raw_suffix"] == ".rw2"
    assert calls["path"].endswith("source.rw2")
    assert calls["postprocess"] == {
        "use_camera_wb": True,
        "output_color": "srgb",
        "output_bps": 16,
        "no_auto_bright": True,
        "half_size": False,
    }


def test_read_image_decodes_arw_with_rawpy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_rawpy(monkeypatch, _dark_raw_fixture())

    image = read_image("sony_source.arw")

    assert image.pixels.dtype == np.uint16
    assert image.metadata["raw_suffix"] == ".arw"
    assert image.metadata["no_auto_bright"] is True
    assert calls["path"].endswith("sony_source.arw")


def test_read_image_can_decode_fast_raw_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_rawpy(monkeypatch, _dark_raw_fixture().astype(np.uint8))

    image = read_image("fast_source.arw", raw_half_size=True, raw_output_bps=8)

    assert image.pixels.dtype == np.uint8
    assert image.metadata["raw_suffix"] == ".arw"
    assert image.metadata["requested_output_bps"] == 8
    assert image.metadata["half_size"] is True
    assert calls["postprocess"] == {
        "use_camera_wb": True,
        "output_color": "srgb",
        "output_bps": 8,
        "no_auto_bright": True,
        "half_size": True,
    }


def test_white_dust_detector_accepts_fff_via_rawpy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_rawpy(monkeypatch, _dark_raw_fixture())
    output_dir = ROOT / "test_outputs" / "raw_white_dust"

    manifest = run_white_dust_detector(
        "inspection.fff",
        output_dir,
        WhiteDustConfig(
            detection_long_edge=96,
            local_radius=4,
            max_area=1200,
            max_dim=40,
            max_thickness=16,
        ),
    )

    assert manifest["source_mode"] == "rgb"
    assert manifest["source_metadata"]["raw_suffix"] == ".fff"
    assert manifest["source_metadata"]["requested_output_bps"] == 8
    assert manifest["source_metadata"]["half_size"] is True
    assert manifest["raw_fast_proxy"] is True
    assert manifest["mask_output_mode"] == "legacy_plus_xmp"
    assert manifest["proxy_mask"] is True
    assert manifest["mask_shape"] == manifest["detection_shape"]
    assert manifest["final_mask_pixels"] > 0
    assert Path(manifest["artifacts"]["mask"]).exists()
    assert Path(manifest["artifacts"]["xmp"]).exists()
    assert Path(manifest["artifacts"]["named_xmp"]).exists()
    assert manifest["xmp"]["format"] == "dust-mask-repair-xmp-v1"
    assert manifest["xmp"]["output_mode"] == "legacy_plus_xmp"
    assert calls["postprocess"]["output_bps"] == 8
    assert calls["postprocess"]["half_size"] is True
