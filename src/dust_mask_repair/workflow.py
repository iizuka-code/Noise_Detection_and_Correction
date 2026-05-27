from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RepairConfig
from .red_highlight import RedHighlightConfig, RedHighlightSourceResult, detect_red_highlight_source_image
from .repair import RepairResult, repair_image


@dataclass(frozen=True)
class RedHighlightRepairResult:
    red_highlight: RedHighlightSourceResult
    repair: RepairResult

    @property
    def generated_mask(self) -> np.ndarray:
        return self.red_highlight.mask

    @property
    def repaired_image(self) -> np.ndarray:
        return self.repair.repaired_image


def repair_image_from_red_highlight(
    image: np.ndarray,
    red_image: np.ndarray,
    red_config: RedHighlightConfig | None = None,
    repair_config: RepairConfig | None = None,
) -> RedHighlightRepairResult:
    normal = np.asarray(image)
    red = np.asarray(red_image)
    _validate_pair_dimensions(normal, red)

    red_result = detect_red_highlight_source_image(red, red_config or RedHighlightConfig())
    repair_cfg = repair_config or RepairConfig(mask_channel="grayscale")
    repair_result = repair_image(normal, red_result.mask, repair_cfg)
    return RedHighlightRepairResult(red_highlight=red_result, repair=repair_result)


def _validate_pair_dimensions(image: np.ndarray, red_image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"image must be an RGB or RGBA array, got shape {image.shape}")
    if red_image.ndim < 2:
        raise ValueError(f"red_image must be an image array, got shape {red_image.shape}")
    if image.shape[:2] != red_image.shape[:2]:
        raise ValueError(
            "image and red-image dimensions differ: "
            f"image={image.shape[1]}x{image.shape[0]}, "
            f"red_image={red_image.shape[1]}x{red_image.shape[0]}"
        )
