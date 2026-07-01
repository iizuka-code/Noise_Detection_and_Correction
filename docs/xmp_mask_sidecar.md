# XMP Mask Sidecar

## Scope

The current XMP output is a host-neutral interchange file for Dust Mask Repair.
It is not an Adobe Camera Raw or Lightroom native local-adjustment mask.

The goal is to keep the detected mask and the metadata needed to interpret it in
one sidecar file, while keeping the existing PNG and JSON outputs available for
visual review and regression testing.

## Files

White-dust detection writes:

- `white_dust_mask.png`: black/white visual mask.
- `white_dust_mask.xmp`: XMP sidecar carrying the same mask as base64 PNG.
- `manifest.json`: detector parameters, image metadata, timings, and component counts.

White-dust repair writes one additional target-sized sidecar:

- `fitted_repair_mask.png`: mask after aspect-aware crop and resize to the JPEG target.
- `fitted_repair_mask.xmp`: sidecar carrying that target-sized mask and the mask-fitting metadata.

The standalone detector also writes:

- `mask.xmp`
- `<source_stem>_white_dust_mask.xmp`

## XMP Namespace

The custom namespace is:

```text
https://dust-mask-repair.local/ns/1.0/
```

Main fields:

- `dmr:format`: currently `dust-mask-repair-xmp-v1`.
- `dmr:role`: `white_dust_detection` or `white_dust_fitted_repair`.
- `dmr:outputMode`: currently `legacy_plus_xmp`, meaning existing PNG/JPEG/JSON outputs remain available and XMP is added alongside them.
- `dmr:maskWidth`, `dmr:maskHeight`, `dmr:maskPixels`.
- `dmr:sourceFileName`, `dmr:targetFileName`.
- `dmr:MaskPngEncoding`: currently `base64/png`.
- `dmr:MaskPng`: the black/white mask PNG bytes encoded as base64 text.
- `dmr:ManifestJson`: detector manifest JSON.
- `dmr:MaskMappingJson`: crop/resize mapping for target-fitted masks.

## Design Notes

- The mask payload is binary PNG so a later importer can reconstruct exactly the
  same 8-bit black/white mask without relying on path-relative external files.
- JSON is duplicated inside XMP so the sidecar remains useful if `manifest.json`
  is moved or lost.
- `read_mask_xmp()` can read this project's sidecars and SilverTurn
  `silverturn-dust-mask-xmp-v1` sidecars back into an 8-bit binary mask.
- Adobe-native XMP compatibility is left for a later adapter slice because the
  local-mask schema is host-specific and should be validated against real target
  application behavior before writing files that claim compatibility.
