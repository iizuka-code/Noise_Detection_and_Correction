from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPAIR_METHODS = (
    "linear",
    "kl",
    "defect_aware",
    "median",
    "inpaint",
    "denoise",
    "hybrid",
    "adaptive",
    "aggressive",
    "wide_scratch",
)
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
    grain_reinject_strength: float = 0.25
    grain_context_radius: int = 8
    grain_blur_radius: int = 1
    grain_min_context_pixels: int = 16
    color_match_strength: float = 0.0
    color_match_radius: int = 8
    color_match_min_context_pixels: int = 12
    edge_guided_enabled: bool = True
    edge_guided_max_component_area: int = 64
    edge_guided_context_radius: int = 4
    edge_guided_search_radius: int = 8
    edge_guided_min_coherence: float = 0.35
    edge_guided_min_gradient_energy: float = 2.5e-4
    edge_guided_max_roi_area: int = 4096
    edge_guided_max_total_search: int = 4096
    tone_guided_enabled: bool = True
    tone_guided_max_component_area: int = 64
    tone_guided_max_roi_area: int = 4096
    tone_guided_context_radius: int = 6
    tone_guided_search_radius: int = 10
    tone_guided_patch_radius: int = 1
    tone_guided_candidate_cap: int = 256
    tone_guided_top_k: int = 5
    tone_guided_tone_weight: float = 2.0
    tone_guided_spatial_weight: float = 0.35
    tone_guided_texture_weight: float = 0.25
    tone_guided_gradient_weight: float = 0.15
    tone_guided_min_context_pixels: int = 8
    defect_core_full_replace: bool = True
    frequency_guided_enabled: bool = False
    frequency_context_radius: int = 16
    frequency_scales: tuple[int, ...] = (1, 2, 4)
    frequency_max_selected_regions: int = 64
    frequency_max_component_area: int = 512
    frequency_max_roi_side: int = 128
    frequency_max_roi_pixels: int = 16384
    frequency_search_radius: int = 36
    frequency_patch_radius: int = 2
    frequency_candidate_cap: int = 128
    frequency_top_k: int = 3
    frequency_min_context_pixels: int = 12
    frequency_min_known_fraction: float = 0.25
    frequency_smooth_threshold: float = 0.0015
    frequency_anisotropy_threshold: float = 0.38
    frequency_band_weight: float = 1.0
    frequency_orientation_weight: float = 0.35
    frequency_border_weight: float = 2.0
    frequency_spatial_weight: float = 0.20
    frequency_texture_weight: float = 0.50
    frequency_midband_strength: float = 0.25

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
        if self.grain_reinject_strength < 0.0:
            raise ValueError("grain_reinject_strength must be >= 0")
        if self.grain_context_radius < 0:
            raise ValueError("grain_context_radius must be >= 0")
        if self.grain_blur_radius < 0:
            raise ValueError("grain_blur_radius must be >= 0")
        if self.grain_min_context_pixels < 0:
            raise ValueError("grain_min_context_pixels must be >= 0")
        if not 0.0 <= self.color_match_strength <= 1.0:
            raise ValueError("color_match_strength must be in the range 0.0..1.0")
        if self.color_match_radius < 0:
            raise ValueError("color_match_radius must be >= 0")
        if self.color_match_min_context_pixels < 0:
            raise ValueError("color_match_min_context_pixels must be >= 0")
        if self.edge_guided_max_component_area < 0:
            raise ValueError("edge_guided_max_component_area must be >= 0")
        if self.edge_guided_context_radius < 0:
            raise ValueError("edge_guided_context_radius must be >= 0")
        if self.edge_guided_search_radius < 0:
            raise ValueError("edge_guided_search_radius must be >= 0")
        if self.edge_guided_min_coherence < 0.0:
            raise ValueError("edge_guided_min_coherence must be >= 0")
        if self.edge_guided_min_gradient_energy < 0.0:
            raise ValueError("edge_guided_min_gradient_energy must be >= 0")
        if self.edge_guided_max_roi_area < 0:
            raise ValueError("edge_guided_max_roi_area must be >= 0")
        if self.edge_guided_max_total_search < 0:
            raise ValueError("edge_guided_max_total_search must be >= 0")
        if self.tone_guided_max_component_area < 0:
            raise ValueError("tone_guided_max_component_area must be >= 0")
        if self.tone_guided_max_roi_area < 0:
            raise ValueError("tone_guided_max_roi_area must be >= 0")
        if self.tone_guided_context_radius < 0:
            raise ValueError("tone_guided_context_radius must be >= 0")
        if self.tone_guided_search_radius < 0:
            raise ValueError("tone_guided_search_radius must be >= 0")
        if self.tone_guided_patch_radius < 0:
            raise ValueError("tone_guided_patch_radius must be >= 0")
        if self.tone_guided_candidate_cap < 0:
            raise ValueError("tone_guided_candidate_cap must be >= 0")
        if self.tone_guided_top_k < 0:
            raise ValueError("tone_guided_top_k must be >= 0")
        if self.tone_guided_tone_weight < 0.0:
            raise ValueError("tone_guided_tone_weight must be >= 0")
        if self.tone_guided_spatial_weight < 0.0:
            raise ValueError("tone_guided_spatial_weight must be >= 0")
        if self.tone_guided_texture_weight < 0.0:
            raise ValueError("tone_guided_texture_weight must be >= 0")
        if self.tone_guided_gradient_weight < 0.0:
            raise ValueError("tone_guided_gradient_weight must be >= 0")
        if self.tone_guided_min_context_pixels < 0:
            raise ValueError("tone_guided_min_context_pixels must be >= 0")
        if self.frequency_context_radius < 0:
            raise ValueError("frequency_context_radius must be >= 0")
        if not self.frequency_scales:
            raise ValueError("frequency_scales must not be empty")
        if any(int(scale) <= 0 for scale in self.frequency_scales):
            raise ValueError("frequency_scales must contain positive integers")
        if self.frequency_max_selected_regions < 0:
            raise ValueError("frequency_max_selected_regions must be >= 0")
        if self.frequency_max_component_area < 0:
            raise ValueError("frequency_max_component_area must be >= 0")
        if self.frequency_max_roi_side < 1:
            raise ValueError("frequency_max_roi_side must be >= 1")
        if self.frequency_max_roi_pixels < 1:
            raise ValueError("frequency_max_roi_pixels must be >= 1")
        if self.frequency_search_radius < 0:
            raise ValueError("frequency_search_radius must be >= 0")
        if self.frequency_patch_radius < 0:
            raise ValueError("frequency_patch_radius must be >= 0")
        if self.frequency_candidate_cap < 0:
            raise ValueError("frequency_candidate_cap must be >= 0")
        if self.frequency_top_k < 0:
            raise ValueError("frequency_top_k must be >= 0")
        if self.frequency_min_context_pixels < 0:
            raise ValueError("frequency_min_context_pixels must be >= 0")
        if not 0.0 <= self.frequency_min_known_fraction <= 1.0:
            raise ValueError("frequency_min_known_fraction must be in the range 0.0..1.0")
        if self.frequency_smooth_threshold < 0.0:
            raise ValueError("frequency_smooth_threshold must be >= 0")
        if self.frequency_anisotropy_threshold < 0.0:
            raise ValueError("frequency_anisotropy_threshold must be >= 0")
        if self.frequency_band_weight < 0.0:
            raise ValueError("frequency_band_weight must be >= 0")
        if self.frequency_orientation_weight < 0.0:
            raise ValueError("frequency_orientation_weight must be >= 0")
        if self.frequency_border_weight < 0.0:
            raise ValueError("frequency_border_weight must be >= 0")
        if self.frequency_spatial_weight < 0.0:
            raise ValueError("frequency_spatial_weight must be >= 0")
        if self.frequency_texture_weight < 0.0:
            raise ValueError("frequency_texture_weight must be >= 0")
        if self.frequency_midband_strength < 0.0:
            raise ValueError("frequency_midband_strength must be >= 0")

