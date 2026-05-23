# Examples

Basic CLI use:

```powershell
dust-mask-repair `
  --image scan.png `
  --mask laser_dust_mask.png `
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

The mask must already be aligned to the scan image. This project does not detect dust and does not align the mask.
