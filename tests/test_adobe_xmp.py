from __future__ import annotations

import pytest

from dust_mask_repair import adobe_native_mask_support, require_adobe_native_mask_support


def test_adobe_native_mask_adapter_is_explicitly_not_enabled() -> None:
    status = adobe_native_mask_support()

    assert status.supported is False
    assert "host-neutral XMP" in status.reason
    assert any("Lightroom/Camera Raw XMP" in item for item in status.required_inputs)


def test_adobe_native_mask_adapter_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="Required before enabling"):
        require_adobe_native_mask_support()
