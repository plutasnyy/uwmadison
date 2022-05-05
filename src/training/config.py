import cv2
import torch

import albumentations as A


class CFG:
    seed = 101
    debug = False  # set debug=False for Full Training
    exp_name = "Baselinev2"
    comment = "unet-efficientnet_b1-224x224-aug2-split2"
    model_name = "Unet"
    backbone = "efficientnet-b1"
    batch_size = 128
    img_size = [224, 224]
    epochs = 15
    lr = 2e-3
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = int(30000 / batch_size * epochs) + 50
    T_0 = 25
    warmup_epochs = 0
    wd = 1e-6
    n_accumulate = max(1, 32 // batch_size)
    n_fold = 5
    num_classes = 3
    precision = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DATA_TRANSFORMS = {
    "train": A.Compose(
        [
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
                    # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                ],
                p=0.25,
            ),
            # A.CoarseDropout(
            #     max_holes=8,
            #     max_height=CFG.img_size[0] // 20,
            #     max_width=CFG.img_size[1] // 20,
            #     min_holes=5,
            #     fill_value=0,
            #     mask_fill_value=0,
            #     p=0.5,
            # ),
            # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        p=1.0,
    ),
    "valid": A.Compose(
        [
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ],
        p=1.0,
    ),
}
