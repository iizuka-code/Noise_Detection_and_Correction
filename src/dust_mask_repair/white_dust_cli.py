from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import REPAIR_METHODS, RepairConfig
from .io import read_image, write_image
from .repair import repair_image
from .white_dust import WhiteDustConfig, detect_white_dust_source_image, run_white_dust_detector


def build_detect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dust-mask-detect-white",
        description="Detect a black/white mask from a white-dust inspection image.",
    )
    parser.add_argument("--source", required=True, help="White-dust inspection image.")
    parser.add_argument("--output-dir", required=True, help="Directory for mask, overlay, and manifest outputs.")
    _add_white_detection_args(parser)
    return parser


def build_repair_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dust-mask-repair-white",
        description="Detect a white-dust mask from an inspection image and repair a target image.",
    )
    parser.add_argument("--image", required=True, help="Target RGB/RGBA image to repair.")
    parser.add_argument("--source", required=True, help="White-dust inspection image used to generate the mask.")
    parser.add_argument("--output", required=True, help="Repaired image output path.")
    parser.add_argument("--mask-output", default=None, help="Optional generated mask output path.")
    parser.add_argument("--method", choices=REPAIR_METHODS, default="kl")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dilate-radius", type=int, default=0)
    parser.add_argument("--feather-radius", type=int, default=0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--max-component-area", type=int, default=5000)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--frequency-scope-mask", default=None, help="Optional black/white mask limiting frequency-guided repair to selected defects.")
    parser.add_argument("--frequency-guided", action="store_true", help="Enable selected local frequency-guided repair inside method=defect_aware.")
    _add_white_detection_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_detect_parser().parse_args(argv)
    manifest = run_white_dust_detector(
        args.source,
        args.output_dir,
        _white_config_from_args(args, visual_artifacts=True),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def repair_main(argv: list[str] | None = None) -> int:
    args = build_repair_parser().parse_args(argv)
    image = read_image(args.image)
    source = read_image(args.source)
    frequency_scope_pixels = read_image(args.frequency_scope_mask).pixels if args.frequency_scope_mask else None
    if image.pixels.shape[:2] != source.pixels.shape[:2]:
        raise ValueError(
            "image and source dimensions differ: "
            f"image={image.pixels.shape[1]}x{image.pixels.shape[0]}, "
            f"source={source.pixels.shape[1]}x{source.pixels.shape[0]}"
        )

    dust_result = detect_white_dust_source_image(
        source.pixels,
        _white_config_from_args(args, visual_artifacts=True),
        source_overlay=True,
    )
    repair_cfg = RepairConfig(
        method=args.method,
        mask_channel="grayscale",
        threshold=args.threshold,
        dilate_radius=args.dilate_radius,
        feather_radius=args.feather_radius,
        strength=args.strength,
        min_component_area=args.min_component_area,
        max_component_area=args.max_component_area,
        padding=args.padding,
        debug_dir=args.debug_dir,
        frequency_guided_enabled=bool(args.frequency_guided or args.frequency_scope_mask),
    )
    repair_result = repair_image(image.pixels, dust_result.mask, repair_cfg, frequency_scope_mask=frequency_scope_pixels)
    write_image(Path(args.output), repair_result.repaired_image)
    if args.mask_output:
        write_image(Path(args.mask_output), dust_result.mask)

    response = {
        "detector": dust_result.manifest,
        "repair": repair_result.metrics,
        "output": str(Path(args.output)),
        "mask_output": str(Path(args.mask_output)) if args.mask_output else None,
    }
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0


def _add_white_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detection-long-edge", type=int, default=1024)
    parser.add_argument("--local-radius", type=int, default=0)
    parser.add_argument("--background-mode", choices=("dark", "brown", "any"), default="dark")
    parser.add_argument("--mask-edge-mode", choices=("tight", "normal", "wide"), default="normal")
    parser.add_argument("--threshold-sensitivity", type=float, default=1.0)
    parser.add_argument("--whiteness-min", type=float, default=0.60)
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--max-area", type=int, default=12000)
    parser.add_argument("--max-dim", type=int, default=900)
    parser.add_argument("--max-thickness", type=float, default=54.0)
    parser.add_argument("--focus-margin-x", type=float, default=0.0)
    parser.add_argument("--focus-margin-y", type=float, default=0.0)


def _white_config_from_args(args: argparse.Namespace, *, visual_artifacts: bool) -> WhiteDustConfig:
    return WhiteDustConfig(
        detection_long_edge=args.detection_long_edge,
        local_radius=args.local_radius,
        background_mode=args.background_mode,
        mask_edge_mode=args.mask_edge_mode,
        threshold_sensitivity=args.threshold_sensitivity,
        whiteness_min=args.whiteness_min,
        min_area=args.min_area,
        max_area=args.max_area,
        max_dim=args.max_dim,
        max_thickness=args.max_thickness,
        focus_margin_x=args.focus_margin_x,
        focus_margin_y=args.focus_margin_y,
        visual_artifacts=visual_artifacts,
    )


if __name__ == "__main__":
    raise SystemExit(main())
