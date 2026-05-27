from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPAIR_METHODS = ("median", "inpaint", "denoise", "hybrid", "aggressive", "wide_scratch")
MASK_CHANNELS = ("auto", "grayscale", "alpha", "red", "max_rgb")


@dataclass(frozen=True)
class RepairConfig:
    method: str = "hybrid"
    mask_channel: str = "auto"
    threshold: float = 0.5
    dilate_radius: int = 2
    feather_radius: int = 2
    strength: float = 1.0
    min_component_area: int = 1
    max_component_area: int | None = 5000
    padding: int = 16
    debug_dir: str | Path | None = None
    collect_debug_images: bool = False

    def validate(self) -> None:
        if self.method not in REPAIR_METHODS:
            raise ValueError(f"Unsupported repair method: {self.method}")
        if self.mask_channel not in MASK_CHANNELS:
            raise ValueError(f"Unsupported mask channel: {self.mask_channel}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in the range 0.0..1.0")
        if self.dilate_radius < 0:
            raise ValueError("dilate_radius must be >= 0")
        if self.feather_radius < 0:
            raise ValueError("feather_radius must be >= 0")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in the range 0.0..1.0")
        if self.min_component_area < 0:
            raise ValueError("min_component_area must be >= 0")
        if self.max_component_area is not None and self.max_component_area < self.min_component_area:
            raise ValueError("max_component_area must be >= min_component_area")
        if self.padding < 0:
            raise ValueError("padding must be >= 0")
