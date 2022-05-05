import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def load_img(path, extend=False):
    """
    Returns 2 dimensional np.array with uint8 in 0-255 range
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype("float32")  # original is uint16
    img = (
        (img - img.min()) / (img.max() - img.min()) * 255.0
    )  # scale image to [0, 255] # TODO instead of max use 15865 of 65535 / 32767
    img = img.astype("uint8")
    return img


def load_3c_float_img(path):
    """
    Returns 3 dimensional np.array with floats in 0-1 range
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    img = img.astype("float32")  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk


def show_img(img, mask=None):
    plt.imshow(img, cmap="bone")

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [
            Rectangle((0, 0), 1, 1, color=_c)
            for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis("off")


def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(5 * 5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx + 1)

        img = (
            imgs[
                idx,
            ]
            .permute((1, 2, 0))
            .numpy()
            * 255.0
        )
        img = img.astype("uint8")

        msk = (
            msks[
                idx,
            ]
            .permute((1, 2, 0))
            .numpy()
            * 255.0
        )
        msk = msk.astype("uint8")

        show_img(img, msk)
    plt.tight_layout()
    plt.show()
