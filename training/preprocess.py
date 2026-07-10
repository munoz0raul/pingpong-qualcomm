#!/usr/bin/env python3
"""
Pre-processing of the data for Phase A (1-box detection).

What it does, for each split (training/ and testing/):
  1. reads the images + boxes via labels.py
  2. resizes every image to IMG_SIZE x IMG_SIZE (320^2) — fixed resolution,
     because the network needs a constant-size input and the dataset has
     very mixed resolutions (320^2 up to 4160x6240).
  3. converts the box to the NORMALIZED target of the network:
        [obj, cx, cy, w, h]
     where obj = 1 if there is a paddle (0 = background) and cx,cy,w,h are
     fractions from 0..1.
  4. saves a compact cache: images.npy + targets.npy

Why save a .npy cache?  Decoding hundreds of JPEGs (some 24 MP) every training
epoch is slow. Doing it ONCE and saving uint8, the training loads everything
with mmap and flies. (Same pattern as the Tibia project.)

** The trick of the normalized coordinates **
The original box is in pixels of the ORIGINAL image. If we divide by the
original size (cx = (x + w/2) / W_orig, etc.), we get fractions 0..1. These
fractions do NOT change when we resize the image — 30% of the width stays 30%
after the resize. That is why we do not need to track a scale factor:
normalizing already solves it, and the image resize and the box resize become
automatically consistent.

Usage:
    training/.venv/bin/python training/preprocess.py           # both splits
    training/.venv/bin/python training/preprocess.py --force   # rebuilds the cache
"""

import argparse
import os

import cv2
import numpy as np

from labels import load_split

IMG_SIZE = 320                     # square network input (H = W = 320)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_THIS_DIR, "cache")

# Target layout (vector of 5 floats per image):
#   index 0 = obj  (1.0 has paddle / 0.0 background)
#   index 1 = cx   (center x, fraction 0..1)
#   index 2 = cy   (center y, fraction 0..1)
#   index 3 = w    (width, fraction 0..1)
#   index 4 = h    (height, fraction 0..1)
TARGET_DIM = 5


def box_to_target(box, img_w, img_h):
    """Box in absolute pixels -> [1, cx, cy, w, h] normalized (fractions 0..1)."""
    cx = (box.x + box.w / 2.0) / img_w
    cy = (box.y + box.h / 2.0) / img_h
    w = box.w / img_w
    h = box.h / img_h
    # defensive clamp: some annotated boxes touch/cross the border
    cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in (cx, cy, w, h))
    return np.array([1.0, cx, cy, w, h], dtype=np.float32)


def build_cache(split, force=False):
    img_out = os.path.join(CACHE_DIR, f"{split}_images.npy")
    tgt_out = os.path.join(CACHE_DIR, f"{split}_targets.npy")
    if os.path.exists(img_out) and os.path.exists(tgt_out) and not force:
        print(f"[cache] {split}: already exists (use --force to rebuild)")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    samples = load_split(split)
    n = len(samples)

    images = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    targets = np.zeros((n, TARGET_DIM), dtype=np.float32)  # all 0 = background by default

    for i, s in enumerate(samples):
        bgr = cv2.imread(s.path)          # OpenCV reads in BGR
        if bgr is None:
            print(f"  warning: could not read {s.path}, skipping")
            continue
        h0, w0 = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        images[i] = resized

        box = s.main_box                  # largest box, or None if background
        if box is not None:
            targets[i] = box_to_target(box, w0, h0)

        if (i + 1) % 100 == 0:
            print(f"  {split}: {i + 1}/{n}")

    np.save(img_out, images)
    np.save(tgt_out, targets)
    com = int((targets[:, 0] > 0.5).sum())
    print(f"[cache] {split}: {n} images saved  (with paddle: {com}, background: {n - com})")
    print(f"        {img_out}  {images.shape} {images.dtype}")
    print(f"        {tgt_out}  {targets.shape} {targets.dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuilds the cache even if it already exists")
    ap.add_argument("--splits", nargs="+", default=["training", "testing"])
    args = ap.parse_args()
    for split in args.splits:
        build_cache(split, force=args.force)


if __name__ == "__main__":
    main()
