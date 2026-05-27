# Dust Mask Repair

Mask-guided dust and spot repair for film scan images.

This is a standalone Python library and CLI for repairing only the pixels indicated by a dust mask PNG. It is intended for workflows where dust has already been detected by a separate laser-lit capture and exported as a mask. The repair engine still does not detect dust from the normal photo scan.

An optional red-highlight detector is included for workflows where a separate red-lit inspection capture makes dust, debris, or scratches appear as bright red marks. That detector writes a black/white mask PNG that can be used by the repair engine.

## Purpose

The tool treats dust and debris as local occlusion defects, not as ordinary image noise. The repair engine:

- reads a normal RGB/RGBA scan image and a same-size mask PNG;
- normalizes the mask to a float `0.0..1.0` mask;
- filters implausibly small or large connected components;
- optionally dilates and feathers the mask;
- repairs each masked component in a padded ROI;
- blends the candidate repair back only through the soft mask.

Pixels outside the soft mask are forced back to the exact original values.

## Installation

```powershell
py -3.12 -m pip install -e .[dev]
```

Runtime dependencies are intentionally small:

- `numpy`: array processing and repair kernels.
- `Pillow`: TIFF fallback and general image support.

PNG read/write uses a small internal 8/16-bit reader/writer so RGB uint16 PNG data is not reduced to 8-bit by Pillow. For stronger TIFF support, install the optional `tiff` extra:

```powershell
py -3.12 -m pip install -e .[tiff]
```

## CLI

Repair from an existing mask:

```powershell
dust-mask-repair `
  --image input.png `
  --mask dust_mask.png `
  --output repaired.png `
  --method hybrid `
  --mask-channel auto `
  --threshold 0.5 `
  --dilate-radius 2 `
  --feather-radius 2 `
  --strength 1.0 `
  --max-component-area 5000 `
  --debug-dir debug_output
```

Detect a mask from a red-highlight inspection image:

```powershell
dust-mask-detect-red `
  --source red_lit_scan.png `
  --output-dir red_mask_output `
  --detection-long-edge 1920 `
  --mask-edge-mode normal `
  --max-red-area 1400 `
  --max-red-dim 95
```

This writes `mask.png`, `<stem>_red_highlight_mask.png`, `preview_mask.png`, `overlay_preview.png`, `overlay.png`, `component_features.json`, and `manifest.json`.

Detect a red-highlight mask and immediately repair the normal scan:

```powershell
dust-mask-repair-red `
  --image normal_scan.png `
  --red-image red_lit_scan.png `
  --output repaired.png `
  --mask-output generated_mask.png `
  --method hybrid
```

`--image` and `--red-image` must already have matching dimensions. Automatic registration, rotation, crop matching, and perspective alignment are not implemented.

Long red scratches are excluded by default because they have different false-positive risk from point dust. Enable elongated scratch detection explicitly:

```powershell
dust-mask-detect-red `
  --source red_lit_scan.png `
  --output-dir red_mask_output `
  --include-long-scratches `
  --max-scratch-dim 720 `
  --max-scratch-width 48 `
  --max-scratch-area 9000
```

The same `--include-long-scratches` options are also accepted by `dust-mask-repair-red`.

Benchmark red-highlight detection followed by masked repair:

```powershell
dust-mask-benchmark `
  --width 1280 `
  --height 853 `
  --iterations 3 `
  --warmup 1 `
  --output-json benchmark_results/red_highlight_1280.json
```

This generates a deterministic synthetic scan and red-lit inspection image in memory, then reports detection time, repair time, total time, generated mask pixels, changed pixels, and traced peak memory. `peak_traced_memory_bytes` is measured with Python `tracemalloc` and may exclude some native allocations.

## Local HTML Test UI

Run a local server and open the printed URL:

```powershell
py -3.12 -m dust_mask_repair.server --host 127.0.0.1 --port 8765
```

The HTML UI has two input modes:

- `mask PNG`: select the normal image and an existing same-size mask image, then run repair.
- `red highlight`: select the normal image and a same-size red-lit inspection image. The server generates a black/white mask first, then runs repair from that generated mask.

After a run, the UI shows before/after comparison, mask, diff, and metrics. Outputs are written under `web_outputs/`.

For testing the artifact guard that prevents dark stains on clean bright regions, open:

```text
http://127.0.0.1:8765/artifact_guard_test.html
```

Supported methods:

- `median`: replaces masked pixels using ROI context median. Simple and conservative for tiny spots on smooth regions.
- `inpaint`: uses a local diffusion-style fill from surrounding known pixels. It does not use OpenCV and does not quantize 16-bit images to 8-bit.
- `denoise`: applies a small local blur only to masked pixels.
- `hybrid`: uses diffusion fill for small regions, median for larger kept regions, then a very light masked smoothing pass.
- `aggressive`: stronger masked replacement for review/testing. It combines diffusion fill, surrounding-ring median replacement, repeated masked smoothing, and a local artifact guard that rejects changes likely to create dark stains on already-clean bright areas.
- `wide_scratch`: fills broad scratch-like regions by interpolating across the narrow axis of each masked span, then lightly smooths only inside the mask. Use this for long or wide scratches after enabling scratch detection.

## Python API

```python
from dust_mask_repair import (
    RedHighlightConfig,
    RepairConfig,
    detect_red_highlight_mask,
    repair_image,
    repair_image_from_red_highlight,
)

mask_result = detect_red_highlight_mask(red_lit_image, RedHighlightConfig())
mask = mask_result.mask

config = RepairConfig(
    method="hybrid",
    mask_channel="auto",
    threshold=0.5,
    dilate_radius=2,
    feather_radius=2,
    strength=1.0,
    min_component_area=1,
    max_component_area=5000,
    padding=16,
)

result = repair_image(image, mask, config)

repaired = result.repaired_image
binary_mask = result.binary_mask
soft_mask = result.soft_mask
metrics = result.metrics

workflow_result = repair_image_from_red_highlight(
    normal_rgb_or_rgba,
    red_lit_rgb,
    red_config=RedHighlightConfig(),
    repair_config=RepairConfig(mask_channel="grayscale"),
)
generated_mask = workflow_result.generated_mask
repaired_from_red = workflow_result.repaired_image
```

`repair_image_from_red_highlight()` is the intended integration point for a host application that already decoded RAW/DNG into an RGB/RGBA working buffer. This repository still does not decode RAW directly.

`RepairResult` includes:

- `repaired_image`
- `binary_mask`
- `soft_mask`
- `changed_bbox_list`
- `metrics`
- `debug_images` when `collect_debug_images=True` or `debug_dir` is set
- `debug_paths` when `debug_dir` is set

## Mask PNG Specification

The mask image must match the input image width and height. Automatic resizing and automatic alignment are intentionally not implemented in the MVP.

`--mask-channel` supports:

- `auto`
- `grayscale`
- `alpha`
- `red`
- `max_rgb`

`auto` chooses alpha if a useful alpha channel exists, grayscale if RGB channels are identical, red if the red channel clearly dominates, otherwise `max_rgb`.

## Debug Output

When `--debug-dir` is set, the tool writes:

- `normalized_mask.png`
- `binary_mask.png`
- `soft_mask.png`
- `repaired_preview.png`
- `diff_visualization.png`
- `metrics.json`

For Python API callers, `RepairConfig(collect_debug_images=True)` keeps the same debug arrays in `RepairResult.debug_images` without writing them to disk. The default is `False` to avoid extra work and memory use during normal repair.

Metrics include:

- `changed_pixel_count`
- `changed_bbox_count`
- `max_abs_diff_outside_mask`
- `mean_abs_diff_inside_mask`
- `mean_abs_diff_outside_mask`
- `processing_time_ms`

## 8-bit and 16-bit Status

- 8-bit RGB/RGBA PNG: supported.
- 8-bit JPEG input/output: supported through Pillow. JPEG output is lossy and not recommended as a preservation format.
- 16-bit RGB/RGBA PNG: supported by the internal PNG path.
- 8-bit TIFF: supported through Pillow.
- 16-bit RGB/RGBA TIFF: supported when optional `tifffile` is installed. Without `tifffile`, Pillow may not preserve 16-bit RGB TIFF data, so writing 16-bit RGB/RGBA TIFF raises an error.

ICC profiles and most metadata are not preserved in this MVP. The output is pixel-data focused.

## Known Limits

- Normal-scan dust detection is out of scope. The repair engine requires a mask PNG or a separate red-highlight inspection image.
- Red-highlight detection currently supports rendered RGB inputs such as PNG, JPEG, and TIFF. RAW decode is not part of this repository.
- No automatic image/mask registration.
- No default resizing when dimensions differ.
- No global denoise, blur, sharpening, or color correction.
- No generative AI, diffusion model, GAN, or large ML inpainting model.
- The built-in `inpaint` method is a deterministic local fill, not OpenCV Telea/Navier-Stokes.
- Large repair regions are still bounded by `--max-component-area` in the repair stage.
- The red-highlight detector is tuned for red illuminated dust, debris, and short defects by default. Long scratches require explicit `--include-long-scratches` detection settings.

## Future Integration Notes

When integrating with the film negative converter:

- pass decoded RGB/RGBA arrays into `repair_image_from_red_highlight()` rather than adding RAW decode to this repository;
- apply identical geometric transforms to the image and mask;
- crop, rotation, and resize can shift mask coordinates;
- compare applying repair to the scan-stage RGB image versus the inverted RGB image;
- add integration tests that verify mask-outside pixel invariance;
- keep repair as a local masked stage, separate from global noise reduction and color transforms.
