from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdobeNativeMaskSupport:
    supported: bool
    reason: str
    required_inputs: tuple[str, ...]


def adobe_native_mask_support() -> AdobeNativeMaskSupport:
    return AdobeNativeMaskSupport(
        supported=False,
        reason=(
            "Adobe Camera Raw / Lightroom native local masks require host-specific "
            "mask metadata and associated binary mask files/digests. The current "
            "exporter intentionally writes a custom host-neutral XMP sidecar instead."
        ),
        required_inputs=(
            "A real Lightroom/Camera Raw XMP sidecar containing an imported local mask",
            "The associated Adobe-generated binary mask payload, if any",
            "A round-trip validation target and accepted/failed import behavior",
        ),
    )


def require_adobe_native_mask_support() -> None:
    status = adobe_native_mask_support()
    if not status.supported:
        required = "; ".join(status.required_inputs)
        raise NotImplementedError(f"{status.reason} Required before enabling: {required}")
