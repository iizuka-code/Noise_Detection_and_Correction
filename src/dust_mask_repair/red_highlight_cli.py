from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import MASK_CHANNELS, REPAIR_METHODS, RepairConfig
from .io import read_image, write_image
from .red_highlight import RedHighlightConfig, run_red_highlight_detector
from .repair import repair_image
from .workflow import repair_image_from_red_highlight


def build_detect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dust-mask-detect-red",
        description="Detect red-highlighted dust, debris, and optionally long scratches as a black/white mask.",
    )
    _add_red_detection_arguments(parser)
    parser.add_argument("--source", required=True, help="Input red-highlight RGB image.")
    parser.add_argument("--output-dir", required=True, help="Directory for mask.png and diagnostics.")
    return parser


def build_repair_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dust-mask-repair-red",
        description="Detect a red-highlight mask, then repair a normal scan image with that mask.",
    )
    _add_red_detection_arguments(parser)
    parser.add_argument("--image", required=True, help="Input normal RGB/RGBA scan image.")
    parser.add_argument("--red-image", required=True, help="Input red-highlight RGB image used to generate the mask.")
    parser.add_argument("--output", required=True, help="Repaired image output path.")
    parser.add_argument("--mask-output", default=None, help="Optional generated mask output path.")
    parser.add_argument("--red-debug-dir", default=None, help="Optional directory for red-highlight diagnostics.")
    parser.add_argument("--method", choices=REPAIR_METHODS, default="hybrid")
    parser.add_argument("--mask-channel", choices=MASK_CHANNELS, default="grayscale")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dilate-radius", type=int, default=2)
    parser.add_argument("--feather-radius", type=int, default=2)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--max-component-area", type=int, default=5000)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--repair-debug-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_detect_parser().parse_args(argv)
    config = _red_config_from_args(args)
    manifest = run_red_highlight_detector(Path(args.source), Path(args.output_dir), config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def repair_main(argv: list[str] | None = None) -> int:
    args = build_repair_parser().parse_args(argv)
    red_config = _red_config_from_args(args)
    normal_image = read_image(args.image)
    red_image = read_image(args.red_image)
    repair_config = RepairConfig(
        method=args.method,
        mask_channel=args.mask_channel,
        threshold=args.threshold,
        dilate_radius=args.dilate_radius,
        feather_radius=args.feather_radius,
        strength=args.strength,
        min_component_area=args.min_component_area,
        max_component_area=args.max_component_area,
        padding=args.padding,
        debug_dir=args.repair_debug_dir,
    )
    if normal_image.pixels.shape[:2] != red_image.pixels.shape[:2]:
        raise ValueError(
            "image and red-image dimensions differ: "
            f"image={normal_image.pixels.shape[1]}x{normal_image.pixels.shape[0]}, "
            f"red_image={red_image.pixels.shape[1]}x{red_image.pixels.shape[0]}"
        )

    if args.red_debug_dir:
        red_manifest = run_red_highlight_detector(Path(args.red_image), Path(args.red_debug_dir), red_config)
        mask_path = Path(red_manifest["artifacts"]["mask"])
        mask = read_image(mask_path).pixels
        repair_result = repair_image(normal_image.pixels, mask, repair_config)
    else:
        workflow_result = repair_image_from_red_highlight(
            normal_image.pixels,
            red_image.pixels,
            red_config=replace(red_config, visual_artifacts=False),
            repair_config=repair_config,
        )
        mask = workflow_result.generated_mask
        mask_path = Path(args.mask_output) if args.mask_output else Path(args.output).with_name(Path(args.output).stem + "_red_mask.png")
        write_image(mask_path, mask)
        red_manifest = workflow_result.red_highlight.manifest
        repair_result = workflow_result.repair

    if args.mask_output and not Path(args.mask_output).exists():
        write_image(Path(args.mask_output), mask)

    write_image(Path(args.output), repair_result.repaired_image)
    print(
        json.dumps(
            {
                "red_highlight": red_manifest,
                "generated_mask": str(mask_path),
                "repair": repair_result.metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_red_detection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detection-long-edge", type=int, default=1920)
    parser.add_argument("--mask-edge-mode", choices=["tight", "normal", "wide", "legacy"], default="normal")
    parser.add_argument("--disable-full-resolution-refine", action="store_true")
    parser.add_argument("--red-min", type=float)
    parser.add_argument("--red-excess-min", type=float)
    parser.add_argument("--red-ratio-min", type=float)
    parser.add_argument("--contrast-red-min", type=float)
    parser.add_argument("--contrast-excess-min", type=float)
    parser.add_argument("--glow-signal-min", type=float)
    parser.add_argument("--contrast-glow-min", type=float)
    parser.add_argument("--hot-core-value-min", type=float)
    parser.add_argument("--hot-core-contrast-min", type=float)
    parser.add_argument("--threshold-sensitivity", type=float, default=1.0)
    parser.add_argument("--min-red-area", type=int, default=1)
    parser.add_argument("--max-red-area", type=int, default=1400)
    parser.add_argument("--max-red-dim", type=int, default=95)
    parser.add_argument(
        "--include-long-scratches",
        action="store_true",
        help="Also keep thin elongated red components that exceed the normal dust/debris size limits.",
    )
    parser.add_argument("--min-scratch-aspect", type=float, default=5.0)
    parser.add_argument("--max-scratch-area", type=int, default=9000)
    parser.add_argument("--max-scratch-dim", type=int, default=720)
    parser.add_argument("--max-scratch-width", type=int, default=48)
    parser.add_argument("--disable-border-glow-suppression", action="store_true")
    parser.add_argument("--debug-artifacts", action="store_true")


def _red_config_from_args(args: argparse.Namespace) -> RedHighlightConfig:
    return RedHighlightConfig(
        detection_long_edge=int(args.detection_long_edge),
        mask_edge_mode=str(args.mask_edge_mode),
        full_resolution_refine=not bool(args.disable_full_resolution_refine),
        red_min=args.red_min,
        red_excess_min=args.red_excess_min,
        red_ratio_min=args.red_ratio_min,
        contrast_red_min=args.contrast_red_min,
        contrast_excess_min=args.contrast_excess_min,
        glow_signal_min=args.glow_signal_min,
        contrast_glow_min=args.contrast_glow_min,
        hot_core_value_min=args.hot_core_value_min,
        hot_core_contrast_min=args.hot_core_contrast_min,
        threshold_sensitivity=float(args.threshold_sensitivity),
        min_area=int(args.min_red_area),
        max_area=int(args.max_red_area),
        max_dim=int(args.max_red_dim),
        include_long_scratches=bool(args.include_long_scratches),
        min_scratch_aspect=float(args.min_scratch_aspect),
        max_scratch_area=int(args.max_scratch_area),
        max_scratch_dim=int(args.max_scratch_dim),
        max_scratch_width=int(args.max_scratch_width),
        suppress_border_glow=not bool(args.disable_border_glow_suppression),
        debug_artifacts=bool(args.debug_artifacts),
    )


if __name__ == "__main__":
    raise SystemExit(main())
