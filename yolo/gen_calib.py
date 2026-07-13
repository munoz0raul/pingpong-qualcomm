#!/usr/bin/env python3
"""
Generates A16W8/INT8 calibration data for qairt-quantizer from the YOLO dataset.

Why here (on the Mac) and reusing infer_yolo: quantization learns the ranges
(min/max) of each tensor by running the model on REAL images. Those images must
pass through EXACTLY the same pre-processing as inference (letterbox 320, RGB,
/255, NCHW) — otherwise the ranges come out wrong and the quantized model loses
accuracy. We reuse _letterbox from infer_yolo.py to guarantee bit-identity.

Output: a calib/ folder with N .raw files (float32, NCHW layout [1,3,320,320]) and
an input_list.txt that the quantizer consumes on the x86 QAIRT machine.

Important reproducibility detail: by default the input list contains paths like
`calib/calib_0000.raw` (relative to the directory where you run qairt-quantizer),
NOT `/Users/...` absolute paths from the Mac. This makes the folder portable:
copy `calib/` next to `best_fp.dlc`, `cd` to that work directory, and run:

    qairt-quantizer --input_dlc best_fp.dlc --input_list calib/input_list.txt ...

Use `--absolute` only when the quantizer will run on the same machine/path where
this script generated the files.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))
from infer_yolo import _letterbox, IMG_SIZE   # same pre-processing as inference

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(_THIS_DIR, "dataset", "images", "train")
DEFAULT_OUT = os.path.join(_THIS_DIR, "calib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="source images directory (default: yolo/dataset/images/train)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output calibration directory (default: yolo/calib)")
    ap.add_argument("--n", type=int, default=200,
                    help="number of calibration images to sample")
    ap.add_argument("--absolute", action="store_true",
                    help="write absolute paths in input_list.txt (default: portable relative paths)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(args.src, "*.jpg")))
    # uniform sampling to vary scenes/light (not just the first N)
    if len(imgs) > args.n:
        step = len(imgs) / args.n
        imgs = [imgs[int(i * step)] for i in range(args.n)]

    lines = []
    for i, path in enumerate(imgs):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        canvas, _, _, _ = _letterbox(bgr, IMG_SIZE)          # letterbox 320x320
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)   # CHW
        arr = chw[np.newaxis, ...].copy()                    # (1,3,320,320) NCHW
        raw = os.path.join(args.out, f"calib_{i:04d}.raw")
        arr.astype(np.float32).tofile(raw)
        if args.absolute:
            lines.append(os.path.abspath(raw))
        else:
            # Portable path for the common QAIRT work layout:
            #   workdir/best_fp.dlc
            #   workdir/calib/input_list.txt
            #   workdir/calib/calib_0000.raw
            lines.append(os.path.join(os.path.basename(args.out), os.path.basename(raw)))

    input_list = os.path.join(args.out, "input_list.txt")
    with open(input_list, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"generated {len(lines)} .raw files in {args.out}")
    print(f"input_list.txt with {len(lines)} lines -> {input_list}")
    rel_desc = f"portable relative ({os.path.basename(args.out)}/*.raw)"
    print("input_list paths: " + ("absolute" if args.absolute else rel_desc))
    print(f"shape per file: (1,3,{IMG_SIZE},{IMG_SIZE}) float32 = {1*3*IMG_SIZE*IMG_SIZE*4} bytes")


if __name__ == "__main__":
    main()
