"""Downsampling, expanding, Gaussian and Laplacian pyramids, and blending."""

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

def downsample(image):
    """Drop odd rows and columns"""
    return image[::2,  ::2]

def upsample(image):
    height, width = image.shape[:2]
    upsampled = np.zeros((height*2, width*2)) 
    upsampled[::2, ::2] = image
    return upsampled

def expand(image, func=None, **kwargs):
    """Add rows and columns"""
    upsampled = upsample(image)
    if func is None:
        kernel = np.array([[0.25, 0.5, 0.25],
                           [0.5, 1, 0.5],
                           [0.25, 0.5, 0.25]])
        upsampled = cv2.filter2D(upsampled, -1, kernel=kernel)
    else:
        upsampled = func(upsampled, **kwargs)
    return upsampled

def gaussian_pyramid(image, N=5, ksize=(3,3)):
    n = 0
    gs = [image]
    h, w = image.shape[:2]
    while h > 1 and w > 1 and n < N - 1:
        gn_p = cv2.blur(gs[n], ksize=ksize)
        h, w = gn_p.shape[:2]
        gs.append(downsample(gn_p))
        n += 1
    return gs
    
def laplacian_pyramid(gpyramid, ksize=(3,3)):
    Ls = []
    for n in range(len(gpyramid[:-1])):
        Ls.append(gpyramid[n].astype(float) - expand(gpyramid[n+1]).astype(float))
    Ls.append(gpyramid[-1].astype(float))
    return Ls

def blend(image_a, image_b, threshold, N):
    threshold = threshold / 255
    ga = gaussian_pyramid(image_a, N=N)
    gb = gaussian_pyramid(image_b, N=N)
    gt = gaussian_pyramid(threshold, N=N)

    la = laplacian_pyramid(ga)
    lb = laplacian_pyramid(gb)

    Ls = []
    
    for lan, lbn, gtn in zip(la, lb, gt):
        Ls.append(lan*gtn + lbn*(1-gtn))

    rLs = Ls[::-1]
    ln = rLs[0]
    for ln_next in rLs[1:]:
        ln = ln_next + expand(ln)
    return ln
    
# Modified laplacian so we can use it with features
def laplacian_pyramid2(image, ksize=(3,3)):
    Ls = []
    gpyramid = gaussian_pyramid(image)
    for n in range(len(gpyramid[:-1])):
        Ls.append(gpyramid[n].astype(float) - expand(gpyramid[n+1]).astype(float))
    Ls.append(gpyramid[-1].astype(float))
    return Ls

# Suggested by Claude Sonnet 5.5
def rescale_to_uint8(level):
    low, high = level.min(), level.max()
    if high == low:
        return np.zeros(level.shape, dtype=np.uint8)
    return ((level - low) / (high - low) * 255).round().astype(np.uint8)

# Suggested by Claude Sonnet 5.5
def laplacian_pyramid_uint8(image, **kwargs):
    return [rescale_to_uint8(level) for level in laplacian_pyramid2(image, **kwargs)]
