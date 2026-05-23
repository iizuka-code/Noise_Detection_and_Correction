# Dust Mask Repair — Agent Instructions

## Commands

Run tests:

```powershell
py -3.12 -m pytest -q -p no:cacheprovider
```

Run a basic syntax/build check:

```powershell
py -3.12 -m compileall src
```

No type checker or linter is configured yet. If one is added later, document the command here and keep it part of the completion checks.

## Quality Rules

- The most important quality condition in this repository is mask-outside pixel invariance.
- Pixels outside the final soft mask must remain byte-for-byte identical to the input image.
- Do not add global blur, global denoise, global sharpening, or global color correction.
- Repair must be local to mask components and their padded ROI context.
- Do not use generative AI inpainting models, diffusion models, GANs, or large ML models in the MVP.
- Dust detection is out of scope. This repository assumes a mask PNG is already provided.
- Default behavior must error when image and mask dimensions differ.
- Preserve 16-bit data where the current I/O path supports it.
- Use `aggressive` only when the user wants visibly stronger repair; `hybrid` remains the conservative default.
