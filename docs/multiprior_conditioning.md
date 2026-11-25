# Geometric-prior conditioning on domain A

`--dataset_mode multiprior_unaligned` lets domain A (the simulated side) carry geometric
prior buffers next to the grayscale render, as extra generator **input** channels. The
translation direction and output are unchanged: the generator still outputs a 3-channel
image and the discriminator only ever sees that image, so priors improve the sim→real
rendering without the model ever having to *predict* them (inference is sim→real, and the
simulation always provides the priors).

## Dataset layout
Produced by `extract_dataset_multiprior.py` in the poseforge repo:

```
<dataroot>/
  trainA/                image (grayscale render)      .jpg
  trainA_shading/        shading prior                 .png   (same stems)
  trainA_depth/          depth prior                   .png
  trainA_normals/        normals prior                 .png
  trainA_segmentation/   segmentation prior            .png
  trainA_mask/           silhouette (for --lambda_edge).png
  trainB/                real images (no priors)       .jpg
  testA/ testA_*/ testB/ validation split, same layout
```

## The flag
`--domainA_buffers` is a comma list over `{image, shading, depth, normals, segmentation}`.
Every buffer is loaded as 3-channel RGB and receives the **same** random crop/flip/zoom as
the image. `input_nc` is derived automatically (`3 + 3 × #priors`); leave it unset.

| Mode | `--domainA_buffers` | `input_nc` |
|------|---------------------|-----------|
| grayscale only (identical to stock CUT) | `image` | 3 |
| grayscale + all priors | `image,shading,depth,normals,segmentation` | 15 |
| priors only (no grayscale) | `shading,depth,normals,segmentation` | 15 (zero image slot) |

## Example commands

Grayscale only (baseline):
```
python train.py --dataroot <dataroot> --name run_gray \
    --dataset_mode multiprior_unaligned --domainA_buffers image
```

Grayscale + all priors (optionally with the background edge loss):
```
python train.py --dataroot <dataroot> --name run_allpriors \
    --dataset_mode multiprior_unaligned \
    --domainA_buffers image,shading,depth,normals,segmentation \
    --lambda_edge 1.0
```

Priors only:
```
python train.py --dataroot <dataroot> --name run_priorsonly \
    --dataset_mode multiprior_unaligned \
    --domainA_buffers shading,depth,normals,segmentation
```

## How the channels are handled (why training stays stable)
The generator input is always `[image_slot (3 ch), prior channels...]`:
- **real_A**: image_slot = grayscale render (zeros in priors-only mode), priors = real.
- **real_B**: image_slot = the real image, prior channels = zeros (no geometry for real flies).
- **fake_B / idt_B** (generated, 3 ch) are re-padded before the contrastive re-encoding —
  `fake_B` reuses **real_A**'s priors (same frame), `idt_B` uses zeros.

So within every PatchNCE pair the prior channels are identical (real_A↔fake_B both carry
real_A's priors; real_B↔idt_B both zeros), and NCE contrasts only the image content. With
`--domainA_buffers image` the prior channels vanish, `input_nc == output_nc`, and the code
path is byte-for-byte the original CUT.

## Not yet covered
- **Inference** (`inference.py`) must stack the same buffers for the sim input when running a
  prior-trained generator — a parallel change, not included here.
- Auxiliary geometry *prediction* (a decoder head that outputs priors, "let the model try" on
  real data) is a separate, heavier experiment.
