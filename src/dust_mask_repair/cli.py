from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MASK_CHANNELS, REPAIR_METHODS, RepairConfig
from .io import read_image, write_image
from .repair import repair_image
from .xmp import read_mask_xmp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dust-mask-repair",
        description="Mask-guided dust and spot repair for film scan images.",
    )
    parser.add_argument("--image", required=True, help="Input RGB/RGBA PNG or TIFF scan image.")
    parser.add_argument("--mask", required=True, help="Dust mask image or custom XMP sidecar.")
    parser.add_argument("--output", required=True, help="Output PNG or TIFF path.")
    parser.add_argument("--method", choices=REPAIR_METHODS, default="hybrid")
    parser.add_argument("--mask-channel", choices=MASK_CHANNELS, default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dilate-radius", type=int, default=2)
    parser.add_argument("--feather-radius", type=int, default=2)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--max-component-area", type=int, default=5000)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--frequency-scope-mask", default=None, help="Optional black/white mask limiting frequency-guided defect-aware repair to selected defects.")
    parser.add_argument("--frequency-guided", action="store_true", help="Enable selected local frequency-guided repair inside method=defect_aware.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = read_image(args.image)
    mask_path = Path(args.mask)
    mask_pixels = read_mask_xmp(mask_path).mask if mask_path.suffix.lower() == ".xmp" else read_image(mask_path).pixels
    frequency_scope_pixels = read_image(args.frequency_scope_mask).pixels if args.frequency_scope_mask else None
    config = RepairConfig(
        method=args.method,
        mask_channel=args.mask_channel,
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
    result = repair_image(image.pixels, mask_pixels, config, frequency_scope_mask=frequency_scope_pixels)
    write_image(Path(args.output), result.repaired_image)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
