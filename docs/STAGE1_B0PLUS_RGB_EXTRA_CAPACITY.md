# Stage 1 B0+: RGB Extra Capacity Control

## Purpose

B0+ controls for the extra trainable capacity introduced by B1. The fair depth
gain should be reported as `PSNR(B1) - PSNR(B0+)`, in addition to `B1 - B0`.

Every B0+-only change is marked with `Stage-1 B0+` or `B0+ change`.

| File | B0+ change |
|---|---|
| `basicsr/archs/rgb_extra_capacity_mambairv2_arch.py` | Parameter-free RGB luma projection, B1-shaped extra branch, and B1-shaped 1x1 fusion |
| `options/train/mambairv2/train_S1_B0Plus_RGBExtraCapacity_MambaIRv2_x4.yml` | From-scratch 500k training configuration |
| `options/test/mambairv2/test_S1_B0Plus_RGBExtraCapacity_MambaIRv2_x4.yml` | Five-dataset evaluation configuration |

The dataset, MambaIRv2 backbone, reconstruction layers, loss, crop,
augmentation, optimizer, scheduler, batch size, iterations, tiled inference,
and metrics are reused unchanged from Stage0.

## Capacity-matched design

```text
RGB -> original Conv3x3 ---------------------\
                                              concat -> Conv1x1 -> MambaIRv2 -> SR
RGB -> fixed luma projection -> extra Conv3x3 /
```

The fixed luma projection has no trainable parameters. With `embed_dim=174`,
the B0+ additions exactly match B1:

```text
extra Conv3x3: 1 * 174 * 3 * 3 + 174       =  1,740
fusion Conv1x1: 348 * 174 + 174             = 60,726
total addition over B0                      = 62,466

B0 trainable parameters                     = 23,050,713
B0+ trainable parameters                    = 23,113,179
B1 trainable parameters                     = 23,113,179
```

At an LR input size of `H x W`, both B0+ and B1 add exactly
`62,118 * H * W` learned convolution MACs. For the configured 192-pixel HR
training crop (`48 x 48` LR), this is 143,119,872 additional MACs per sample.
B0+ also performs a negligible parameter-free RGB-to-luma calculation.

Because the B0+ extra layers are created in the same order and with the same
shapes as B1, `manual_seed: 10` gives the two controls matching learned-weight
initialization; their auxiliary input information is the intended difference.

## Commands

Full training from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B0Plus_RGBExtraCapacity_MambaIRv2_x4.yml \
  --launcher none
```

Resume an interrupted run:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B0Plus_RGBExtraCapacity_MambaIRv2_x4.yml \
  --launcher none \
  --auto_resume
```

Optional 25% screening run in a separate experiment directory:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B0Plus_RGBExtraCapacity_MambaIRv2_x4.yml \
  --launcher none \
  --force_yml \
    name=S1_B0Plus_RGBExtraCapacity_MambaIRv2_x4_screen25 \
    train:total_iter=125000 \
    train:scheduler:milestones=[62500,100000,112500,118750]
```

Five-dataset test:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
  -opt options/test/mambairv2/test_S1_B0Plus_RGBExtraCapacity_MambaIRv2_x4.yml \
  --launcher none
```
