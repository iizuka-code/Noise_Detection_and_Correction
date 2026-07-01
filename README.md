# Dust Mask Repair

Mask-guided dust and spot repair for film scan images.

This is a standalone Python library, simple GUI, and CLI for repairing only the pixels indicated by a dust mask PNG. It is intended for workflows where dust has already been detected by a separate inspection capture and exported as a mask. The repair engine still does not detect dust from the normal photo scan.

An optional red-highlight detector is included for workflows where a separate red-lit inspection capture makes dust, debris, or scratches appear as bright red marks. That detector writes a black/white mask PNG that can be used by the repair engine.

An optional floating-dust detector is also included for testing inspection images where bright white or colored dust appears over a black/dark base, with a brown-base compatibility mode. This detector writes the same black/white mask format, can export the mask as a custom XMP sidecar, and the local HTML UI can also apply that generated mask to a JPEG repair target.

## Simple GUI

For the main floating-dust workflow, double-click `KLComplementary2_0_GUI.pyw` in this repository, or launch the small desktop GUI after installing the package:

```powershell
dust-mask-repair-gui
```

or run the checkout launcher directly:

```powershell
py -3.12 KLComplementary2_0_GUI.pyw
```

Select:

- `補正対象写真`: the normal photo to repair.
- `マスク作成用写真`: the dark/black-base inspection photo used to generate the mask.
- `補正方式`: `kl`, `linear`, or `defect_aware`.
- `出力フォルダ`: where the run folder should be written.
- `周辺補正 px`: expands the detected mask before repair. The GUI default is `1` px.
- `境界なじませ px`: feathers the expanded repair edge. The GUI default is `2` px.
- `色なじませ強度`: pulls repaired RGB statistics toward the surrounding context. `0.0` disables it; `1.0` is strongest. The GUI default is `0.65`.
- `粒状感強度`: reinjects local grain/noise texture into the repaired area. The GUI default is `0.45`.
- `edge-guidedテスト`: runs a built-in synthetic diagonal-edge micro-dust case without selecting photos, then writes visual outputs and PASS/FAIL metrics to the output folder.

Both `補正対象写真` and `マスク作成用写真` accept rendered image files plus common RAW files including Sony `.arw`. RAW support uses optional `rawpy`; install the raw extra when ARW decode is needed:

```powershell
py -3.12 -m pip install -e .[raw]
```

The GUI writes a timestamped folder containing the repaired image, `target_preview.png`, `generated_mask.png`, `repair_mask_expanded.png`, `inspection_overlay.png`, `white_dust_score.png`, `processing_status.json`, `repair_metrics.json`, and `repair_result.json`. RAW inputs are rendered to RGB for processing; the repaired output is written as a viewable image file, not back into ARW. The target RAW is decoded at 16-bit, while the inspection RAW is decoded at 8-bit full resolution to reduce detector memory without changing mask coordinates. The JSON includes `実行日`, `version`, `対象写真`, `マスク作成用写真`, `補正方法`, and `正答率` (`null` because this workflow has no ground-truth answer image). If processing fails, the GUI writes `error.txt` in the timestamped run folder and `last_error.txt` in the selected output folder.

The `edge-guidedテスト` button writes `edge_guided_clean_answer.png`, `edge_guided_damaged_input.png`, `edge_guided_mask.png`, `edge_guided_repaired_defect_aware.png`, `edge_guided_disabled_comparison.png`, `edge_guided_comparison.png`, `edge_guided_error_heatmap.png`, `edge_guided_test_metrics.json`, and `edge_guided_test_result.json`. The test passes when edge-guided `defect_aware` improves masked MAE over the disabled comparison and leaves pixels outside the mask unchanged. For black dust/line trace regression, the benchmark helper also includes `mottled_background_dark_dust`.

## Purpose

The tool treats dust and debris as local occlusion defects, not as ordinary image noise. The repair engine:

- reads a normal RGB/RGBA scan image and a same-size mask PNG;
- normalizes the mask to a float `0.0..1.0` mask;
- filters implausibly small or large connected components;
- optionally dilates and feathers the mask;
- repairs each masked component in a padded ROI;
- blends the candidate repair back only through the soft mask.

Pixels outside the soft mask are forced back to the exact original values.

For a detailed explanation of the current repair algorithm, see [`docs/repair_algorithm.md`](docs/repair_algorithm.md).
For a current project-wide technical summary, see [`docs/technical_book.md`](docs/technical_book.md).

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

For RAW inspection input such as DNG, ARW, RW2, and FFF, install the optional `raw` extra:

```powershell
py -3.12 -m pip install -e .[raw]
```

RAW support uses `rawpy`/LibRaw to render the source into RGB before detection or repair. `rawpy` auto-brightening is disabled so black-base inspection captures stay dark enough for background-gated mask generation. The floating-dust RAW detector uses a half-size 8-bit proxy and a 1024-pixel default long edge for speed; normal `read_image()` calls still default to full-size 16-bit RAW rendering. The original RAW file is not modified.
`rawpy` is kept optional; its PyPI classifier is MIT, and release packaging should still review the bundled LibRaw terms for the target platform.

## CLI

Repair from an existing mask image or a custom mask XMP sidecar:

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

`--mask` also accepts the custom XMP sidecars written by this project and by SilverTurn's `--dust-mask` flow. Existing PNG/TIFF mask input remains supported.

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

Detect a mask from a white-dust inspection image:

```powershell
dust-mask-detect-white `
  --source dark_base_inspection.png `
  --output-dir white_mask_output `
  --background-mode dark `
  --detection-long-edge 1024
```

Detect a white-dust mask and immediately repair a same-size target image. The default method is `kl`; use `--method defect_aware` for the staged defect-aware router:

```powershell
dust-mask-repair-white `
  --image normal_scan.png `
  --source dark_base_inspection.png `
  --output repaired.png `
  --mask-output generated_mask.png `
  --method defect_aware
```

`--image` and `--source` must already have matching dimensions. The Web UI still has a JPEG convenience path that explicitly fits a generated mask to the target by center crop and nearest-neighbor resize.

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

For the main repair UI, including the two-photo red-highlight workflow, open:

```text
http://127.0.0.1:8765/
```

The HTML UI has two input modes:

- `mask PNG`: select the normal image and an existing same-size mask image, then run repair.
- `red highlight`: select the normal image and a same-size red-lit inspection image. The server generates a black/white mask first, then runs repair from that generated mask.

After a run, the UI shows before/after comparison, mask, diff, and metrics. Outputs are written under `web_outputs/`.
When a RAW file is used as the normal or inspection image, the server renders a preview PNG for browser display while keeping the original upload in the run directory.

For the floating-dust mask-generation and JPEG repair test UI, open:

```text
http://127.0.0.1:8765/white_dust.html
```

Select the black/dark-base inspection image, or switch `background mode` to `brown` for the older brown-base test photos. PNG, JPEG, TIFF, DNG, ARW, RW2, FFF, and other common RAW extensions are accepted when `rawpy` is installed. If only the inspection image is selected, the page calls `/api/detect-white-dust`, generates a black/white mask, writes `white_dust_mask.xmp`, and shows source/overlay split comparison, mask, score map, and detector metrics. If a JPEG repair target is also selected, the page calls `/api/repair-white-dust`, fits the generated mask to the JPEG by aspect-aware center crop and nearest-neighbor resize, writes `fitted_repair_mask.xmp`, then writes `repaired.jpg` and shows before/after repair comparison. RAW sources are decoded as half-size 8-bit inspection proxies and displayed through a generated preview PNG because browsers cannot display RAW files directly. For RAW detection-only runs, the generated mask is preview-sized and matches the displayed source preview. The UI defaults to full-frame dark-background inspection at long edge 1024; raise it only when finer mask detail is worth the extra time.

The XMP sidecar uses the custom `https://dust-mask-repair.local/ns/1.0/` namespace. It embeds the black/white mask as base64 PNG plus the detector manifest and optional mask-fitting metadata. Its `outputMode` is `legacy_plus_xmp`, meaning PNG/JSON/JPEG outputs remain available and XMP is added alongside them. It is a host-neutral interchange file and is not yet an Adobe Camera Raw / Lightroom native local-mask XMP.
See [`docs/xmp_mask_sidecar.md`](docs/xmp_mask_sidecar.md) for the sidecar structure.
See [`docs/adobe_native_xmp_notes.md`](docs/adobe_native_xmp_notes.md) for the native Adobe-mask adapter boundary.

For testing the artifact guard that prevents dark stains on clean bright regions, open:

```text
http://127.0.0.1:8765/artifact_guard_test.html
```


### Selected local frequency-guided repair

`method="defect_aware"` can optionally run a local spatial-frequency guided sub-strategy only on user-selected defects. This is not a new public repair method. It uses a separate black/white `frequency scope mask` to choose which defect components may use the heavier descriptor.

- Unselected defects keep the existing `defect_aware` route.
- If no frequency scope mask is supplied, no frequency descriptor work is performed.
- The scope mask does not expand the editable area; final changes are still limited by the defect repair mask and blend alpha.
- The descriptor is a local multiscale finite-difference/normalized-box-filter approximation, not a full FFT spectrum.

CLI example:

```powershell
py -3.12 -m dust_mask_repair.cli `
  --image target.png `
  --mask defect_mask.png `
  --output repaired.png `
  --method defect_aware `
  --frequency-scope-mask scope.png
```

Python API example:

```python
result = repair_image(
    image,
    defect_mask,
    RepairConfig(method="defect_aware", frequency_guided_enabled=True),
    frequency_scope_mask=scope_mask,
)
```

The desktop GUI has an optional `空間周波数補正範囲マスク` field. When supplied, the run folder includes `frequency_scope_mask.png`, `frequency_selected_core_mask.png`, and `frequency_selected_overlay.png`.

Supported methods:

- `linear`: fills only the masked core by iterative local linear interpolation from surrounding pixels. It is deterministic and keeps pixels outside the final mask unchanged.
- `kl`: starts from the linear fill, then assigns masked pixels so their RGB histogram distribution follows the surrounding context as closely as practical. This is now the default for the white-dust repair workflow.
- `defect_aware`: staged high-quality repair route. It classifies each kept mask component, routes tiny/small/fast/directional/patch defects to dedicated deterministic repairs, uses edge-guided local repair when RGB structure tensor confidence is high, uses tone-guided local repair for small dark dust traces when donor patches are available, fully replaces detected core pixels at `strength=1.0`, optionally reinjects local grain only inside repaired pixels, and falls back to existing kernels for unsupported cases. It is available from CLI/API/GUI/Web; GUI defaults remain `kl`.
- `median`: replaces masked pixels using ROI context median. Simple and conservative for tiny spots on smooth regions.
- `inpaint`: uses a local diffusion-style fill from surrounding known pixels. It does not use OpenCV and does not quantize 16-bit images to 8-bit.
- `denoise`: applies a small local blur only to masked pixels.
- `adaptive`: quality-oriented deterministic repair. It separates core/repair/blend masks, uses normalized-convolution fill, local plane fitting on gradients, optional OpenCV Telea when available for tiny spots, and PCA-guided directional fill for thin diagonal defects.
- `hybrid`: kept for compatibility, but now routes through the adaptive repair kernels instead of switching large regions directly to a single median fill.
- `aggressive`: stronger masked replacement for review/testing. It combines diffusion fill, surrounding-ring median replacement, repeated masked smoothing, and a local artifact guard that rejects changes likely to create dark stains on already-clean bright areas.
- `wide_scratch`: fills broad scratch-like regions by interpolating across the narrow axis of each masked span, then lightly smooths only inside the mask. Use this for long or wide scratches after enabling scratch detection.

## Python API

```python
from dust_mask_repair import (
    RedHighlightConfig,
    RepairConfig,
    WhiteDustConfig,
    detect_red_highlight_mask,
    detect_white_dust_mask,
    repair_image,
    repair_image_from_red_highlight,
    repair_image_from_white_dust,
    write_mask_xmp,
)

mask_result = detect_red_highlight_mask(red_lit_image, RedHighlightConfig())
mask = mask_result.mask

white_result = detect_white_dust_mask(dark_base_image, WhiteDustConfig())
white_mask = white_result.mask

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
    red_config=RedHighlightConfig(visual_artifacts=False),
    repair_config=RepairConfig(mask_channel="grayscale"),
)
generated_mask = workflow_result.generated_mask
repaired_from_red = workflow_result.repaired_image

white_workflow_result = repair_image_from_white_dust(
    normal_rgb_or_rgba,
    dark_base_inspection_rgb,
    white_config=WhiteDustConfig(visual_artifacts=False),
    repair_config=RepairConfig(method="kl", mask_channel="grayscale"),
)
generated_white_mask = white_workflow_result.generated_mask
repaired_from_white = white_workflow_result.repaired_image
```

`detect_white_dust_mask()`, `repair_image_from_red_highlight()`, and `repair_image_from_white_dust()` are the cleanest integration points for a host application that already decoded RAW/DNG into an RGB/RGBA working buffer. For standalone testing, `read_image()` can also decode common RAW files through optional `rawpy` and returns a rendered RGB array.
Use `RedHighlightConfig(visual_artifacts=False)` when the caller only needs the generated mask and repaired pixels; the detector then skips overlay and score-map arrays. The standalone detection CLI and Web UI still generate visual artifacts because they write preview files.
Use `WhiteDustConfig(visual_artifacts=False)` the same way when only the floating-dust mask is needed.

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
- RAW input: supported when optional `rawpy` is installed. DNG, ARW, RW2, FFF, and common camera RAW extensions are rendered to RGB with camera white balance and `no_auto_bright=True` before the existing detector/repair pipeline sees them. The floating-dust detector uses half-size 8-bit RAW proxies to keep interactive testing responsive.

ICC profiles and most metadata are not preserved in this MVP. The output is pixel-data focused.

## Known Limits

- General normal-scan dust detection is out of scope. The repair engine requires a mask PNG or a supported inspection image.
- Red-highlight detection and floating-dust detection support rendered RGB inputs such as PNG, JPEG, and TIFF. RAW input is an optional convenience path via `rawpy`, not a full scene-linear RAW development pipeline.
- No automatic image/mask registration.
- No default resizing when dimensions differ.
- No global denoise, blur, sharpening, or color correction.
- No generative AI, diffusion model, GAN, or large ML inpainting model.
- The built-in `inpaint` method is a deterministic local fill, not OpenCV Telea/Navier-Stokes. The `adaptive` method may use OpenCV Telea only when `cv2` is already available; otherwise it uses the NumPy normalized-convolution fallback.
- Large repair regions are still bounded by `--max-component-area` in the repair stage.
- The red-highlight detector is tuned for red illuminated dust, debris, and short defects by default. Long scratches require explicit `--include-long-scratches` detection settings.
- The floating-dust detector defaults to bright white or colored dust over a black/dark base. Use `WhiteDustConfig(background_mode="brown")` for the older brown-base inspection photos. Photos that include holders, glass edges, or bright frames may need `focus_margin_x`, `focus_margin_y`, and background thresholds.
- When the floating-dust repair UI applies a generated mask to a JPEG with different dimensions or aspect ratio, it uses centered aspect-ratio fitting. It does not perform feature registration or perspective correction.
- XMP mask output is currently a custom host-neutral sidecar. Adobe Camera Raw / Lightroom native local-mask compatibility is a separate adapter task.
- Exact RAW format coverage depends on the LibRaw version bundled with `rawpy`; unsupported camera variants fail with a clear decode error.

## Future Integration Notes

When integrating with the film negative converter:

- pass decoded RGB/RGBA arrays into `repair_image_from_red_highlight()` and `detect_white_dust_mask()` when the host already owns RAW development;
- use this repository's optional RAW decode only for standalone inspection/testing flows where a rendered RGB intermediate is acceptable;
- set `RedHighlightConfig(visual_artifacts=False)` for non-visual batch or export flows;
- set `WhiteDustConfig(visual_artifacts=False)` for non-visual batch or export flows;
- apply identical geometric transforms to the image and mask;
- crop, rotation, and resize can shift mask coordinates;
- compare applying repair to the scan-stage RGB image versus the inverted RGB image;
- add integration tests that verify mask-outside pixel invariance;
- keep repair as a local masked stage, separate from global noise reduction and color transforms.
