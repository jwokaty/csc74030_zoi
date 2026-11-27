"""Loading images and ROI labels, and denoising plate images."""

import cv2
import glob
import numpy as np
import os
import pandas as pd
import re
import requests
import sys
import torch
from pathlib import Path
from csc74030_zoi.paths import FBCNN, PROCESSED

def load_images(file_path, format=re.compile(r"^\d+(?:\.\d+)* [^.]*\.\w+$"), verbose=False, rename=False, column="original"):
    """Read the images in a folder as grayscale arrays.

    Names such as "7.17.1 original.jpg" are rewritten to "7.17.1. original.jpg".
    The files on disk are renamed only when rename=True. The images are stored
    in the table column named by `column`.
    """
    images = pd.DataFrame(columns=["name", column])

    for f in glob.glob(file_path + "/*.*p*g"):
        if verbose:
            print(f"Reading {f}")
        file_name = os.path.basename(f)
        if format.match(file_name):
            fixed_name = re.sub(r"^(\d+(?:\.\d+)*) ", r"\1. ", file_name)
            print(f"Rewriting {file_name} -> {fixed_name}")
            if rename:
                fixed_path = os.path.join(os.path.dirname(f), fixed_name)
                os.rename(f, fixed_path)
                f = fixed_path
            file_name = fixed_name
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        name = re.sub(r'[\s.]', "_", file_name).removesuffix("_jpg")
        images.loc[len(images)] = [name, image]
    return images


# Adapted from main_test_fbcnn_gray.py in FBCNN (https://github.com/jiaxi-jiang/FBCNN),
# licensed under the Apache License 2.0. Changed from the original.
# Note: The following is based on main_test_fbcnn_gray.py with assistance by
# Claude Sonnet 5.5

def denoise_images(images):
    sys.path.insert(0, os.path.dirname(FBCNN))
    from FBCNN.models.network_fbcnn import FBCNN as net

    processed_path = PROCESSED
    weights_name = "fbcnn_gray.pth"
    model_path = os.path.join(FBCNN, "model_zoo", weights_name)
    url = f"https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{weights_name}"

    # download the weights only if they are missing
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(r.content)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = net(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode="R")
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    qf = 50
    qf_t = torch.tensor([[1 - qf / 100]], dtype=torch.float32, device=device)


    def to_uint8(t):
        return np.clip(t.squeeze().cpu().numpy() * 255.0, 0, 255).round().astype(np.uint8)


    def denoise(image):
        x = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)
        with torch.no_grad():
            out, _ = model(x, qf_t)
        return to_uint8(out)


    os.makedirs(processed_path, exist_ok=True)

    images["denoised"] = None
    done, new, skipped = [], [], []
    expected = set()

    for row in images.itertuples():
        if "original" not in row.name or row.original is None:
            skipped.append(row.name)
            continue
        stem = re.sub(r"original.*$", "denoised", row.name)      # 7_17_1__original_jpg -> 7_17_1__denoised
        expected.add(stem)
        path = f"{processed_path}/{stem}.png"

        existing = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
        if existing is not None:                                 # already processed: just load it
            images.at[row.Index, "denoised"] = existing
            done.append(stem)
        else:                                                    # missing or unreadable: denoise it
            result = denoise(row.original)
            if not cv2.imwrite(path, result):
                raise RuntimeError(f"could not write {path}")
            images.at[row.Index, "denoised"] = result
            new.append(stem)

    stray = sorted(p.stem for p in Path(processed_path).glob("*.png") if p.stem not in expected)
    print(f"loaded {len(done)} existing, denoised {len(new)} new: {new}")
    if skipped:
        print("skipped (no 'original' in the name, or no image):", skipped)
    if stray:
        print("files in processed that match no plate:", stray)

    return images
