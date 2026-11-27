"""Global and local thresholding methods."""

import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu

def get_binary_image(image):
    """Custom binary image that groups disks and zones of inhibition together 
    against bacterial lawn"""

    thresholds = threshold_multiotsu(image, classes=3)
    binary_image = np.zeros_like(image)
    binary_image[(image < thresholds[0]) | (image >= thresholds[1])] = 255
    return binary_image

def niblack(x, **kwargs):
    k = kwargs.get("k", -0.2)
    fx = x.ravel()
    return fx.mean() + k * fx.std()

def sauvola(x, **kwargs):
    k = kwargs.get("k", 0.34)
    R = kwargs.get("R", 128)
    fx = x.ravel()
    return fx.mean() * (1 + k * (-1 + fx.std() / R))

def bernsen(x, **kwargs):
    fx = x.ravel()
    return 0.5 * (int(fx.max()) + int(fx.min()))

def local_adapted_threshold(img, size, func=None, **kwargs):
    if size % 2 == 0:
        raise ValueError("size must be odd")

    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    s = size // 2
    off = (size - s) // 2
    for y in range(0, height - size + 1, size //  2):
        for x in range(0, width - size + 1, size // 2):
            win = img[y:y+size, x:x+size]
            T = func(win, **kwargs)
            ys, xs = y + off, x + off
            mask[ys:ys+s, xs:xs+s] = img[ys:ys+s, xs:xs+s] < T
    return mask

## 2 Global Threshold with m = 2, 3 modes

def new_T(img, T1, T2=None):
    if T2 is None:
        g1, g2 = img > T1, img <= T1
        mu1, mu2 = img[g1].ravel().mean(), img[g2].ravel().mean()
        return ((mu1 + mu2) / 2, )
    else:
        g1, g2, g3 = img <= T1, (T1 < img) & (img <= T2), img > T2
        mu1, mu2, mu3 = img[g1].ravel().mean(), img[g2].ravel().mean(), img[g3].ravel().mean()
        return ((mu1 + mu2) / 2, (mu2 + mu3) / 2)

def global_threshold(img, T1, T2=None, limit=10):
    ts = []
    if T2 is None:
        while True:
            t1 = new_T(img, T1)[0]
            if abs(T1 - t1) <= limit:
                break
            T1 = t1
        mask = img > t1
        ts = [math.floor(t1)]
    else:
        while True:
            t1, t2 = new_T(img, T1, T2)
            if abs(T1 - t1) <= limit and abs(T2 - t2) <= limit:
                break
            T1, T2 = t1, t2
        mask = np.zeros_like(img)
        mask[img <= T1] = 0
        mask[(T1 < img) & (img <= T2)] = 1
        mask[img > T2] = 2
        ts = [math.floor(T1), math.floor(T2)]
    return ts, mask

def histo_seg_with_mask(img, low, high, lp_low, lp_high, la_low, la_high, label):
    label_point = (lp_low, lp_high)
    label_at = (la_low, la_high)

    mask = (img > low) & (img <= high)
    counts, edges = np.histogram(img.ravel(), bins=256, range=(0, 256))

    fig, (ax_img, ax_mask, ax_hist) = plt.subplots(1, 3, figsize=(27, 8))

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax_img.imshow(np.ma.masked_where(~mask, mask), alpha=0.5)
    ax_img.annotate(f"peak {label}", xy=label_point, xytext=label_at,
                    arrowprops=dict(arrowstyle="->"))
    ax_img.set_title(f"Image with mask range({low}, {high})")
    
    ax_mask.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax_mask.axis("off")
    ax_mask.set_title(f"Peak {label} Mask")

    ax_hist.bar(edges[:-1], counts, width=1)
    ax_hist.set_xlim(0, 255)
    ax_hist.axvspan(low, high, color="red", alpha=0.3)
    ax_hist.annotate(f"peak {label}", xy=(low, counts[low:high].max()),
                     xytext=(high + 20, counts[low:high].max()),
                     arrowprops=dict(arrowstyle="->"))
    ax_hist.set_title(f"Peak {label} range({low}, {high})")
