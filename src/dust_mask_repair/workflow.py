from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RepairConfig
from .red_highlight import RedHighlightConfig, RedHighlightSourceResult, detect_red_highlight_source_image
from .repair import RepairResult, repair_image
from .white_dust import WhiteDustConfig, WhiteDustSourceResult, detect_white_dust_source_image


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


@dataclass(frozen=True)
class WhiteDustRepairResult:
    white_dust: WhiteDustSourceResult
    repair: RepairResult

    @property
    def generated_mask(self) -> np.ndarray:
        return self.white_dust.mask

    @property
    def repaired_image(self) -> np.ndarray:
        return self.repair.repaired_image


def repair_image_from_red_highlight(
    image: np.ndarray,
    red_image: np.ndarray,
    red_config: RedHighlightConfig | None = None,
    repair_config: RepairConfig | None = None,
    frequency_scope_mask: np.ndarray | None = None,
) -> RedHighlightRepairResult:
    normal = np.asarray(image)
    red = np.asarray(red_image)
    _validate_pair_dimensions(normal, red)

    red_cfg = red_config or RedHighlightConfig(visual_artifacts=False)
    red_result = detect_red_highlight_source_image(red, red_cfg)
    repair_cfg = repair_config or RepairConfig(mask_channel="grayscale")
    repair_result = repair_image(normal, red_result.mask, repair_cfg, frequency_scope_mask=frequency_scope_mask)
    return RedHighlightRepairResult(red_highlight=red_result, repair=repair_result)


def repair_image_from_white_dust(
    image: np.ndarray,
    inspection_image: np.ndarray,
    white_config: WhiteDustConfig | None = None,
    repair_config: RepairConfig | None = None,
    frequency_scope_mask: np.ndarray | None = None,
) -> WhiteDustRepairResult:
    normal = np.asarray(image)
    inspection = np.asarray(inspection_image)
    _validate_pair_dimensions(normal, inspection, inspection_name="inspection_image")

    white_cfg = white_config or WhiteDustConfig(visual_artifacts=False)
    white_result = detect_white_dust_source_image(inspection, white_cfg, source_overlay=False)
    repair_cfg = repair_config or RepairConfig(
        method="kl",
        mask_channel="grayscale",
        threshold=0.5,
        dilate_radius=0,
        feather_radius=0,
    )
    repair_result = repair_image(normal, white_result.mask, repair_cfg, frequency_scope_mask=frequency_scope_mask)
    return WhiteDustRepairResult(white_dust=white_result, repair=repair_result)


def _validate_pair_dimensions(
    image: np.ndarray,
    inspection_image: np.ndarray,
    *,
    inspection_name: str = "red_image",
) -> None:
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"image must be an RGB or RGBA array, got shape {image.shape}")
    if inspection_image.ndim < 2:
        raise ValueError(f"{inspection_name} must be an image array, got shape {inspection_image.shape}")
    if image.shape[:2] != inspection_image.shape[:2]:
        raise ValueError(
            f"image and {inspection_name} dimensions differ: "
            f"image={image.shape[1]}x{image.shape[0]}, "
            f"{inspection_name}={inspection_image.shape[1]}x{inspection_image.shape[0]}"
        )
