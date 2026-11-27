# Note: Some of this file was written by Claude Sonnet 5.5.

"""Locations of data and external files, read from config.toml.

Settings in config.local.toml, if that file exists, replace the ones in
config.toml. Relative paths start at the project root.
"""

import cv2
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(file_name):
    path = PROJECT_ROOT / file_name
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


_paths = _read("config.toml").get("paths", {}) | _read("config.local.toml").get("paths", {})


def _resolve(name):
    # An absolute path in the config replaces the project root
    return (PROJECT_ROOT / _paths[name]).resolve().as_posix()


IMAGES_ORIGINAL = _resolve("images_original")
IMAGES_MEASURED = _resolve("images_measured")
PROCESSED = _resolve("processed")
ROI_LABELS = _resolve("roi_labels")
FBCNN = _resolve("fbcnn")

def get_roi(row, path, h = 1024//4, w = 1024//4):
    file_name = row.roi.split("-")[0]
    image = cv2.imread(f"{path}/{file_name}.png", cv2.IMREAD_GRAYSCALE)
    return image[row.y:row.y+h, row.x:row.x+w]
