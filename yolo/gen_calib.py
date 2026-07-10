#!/usr/bin/env python3
"""
Generates INT8 calibration data for the qairt-quantizer from the YOLO dataset.

Why here (on the Mac) and reusing infer_yolo: INT8 quantization learns the ranges
(min/max) of each tensor by running the model on REAL images. Those images must
pass through EXACTLY the same pre-processing as inference (letterbox 320, RGB,
/255, NCHW) — otherwise the ranges come out wrong and the quantized model loses
accuracy. We reuse _letterbox from infer_yolo.py to guarantee bit-identity.

Output: a calib/ folder with N .raw files (float32, NCHW layout [1,3,320,320]) and
an input_list.txt (one .raw path per line) that the quantizer consumes on the server.
"""
import os
import sys
import glob
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))
from infer_yolo import _letterbox, IMG_SIZE   # same pre-processing as inference

SRC = os.path.join(os.path.dirname(__file__), "dataset", "images", "train")
OUT = os.path.join(os.path.dirname(__file__), "calib")
N = 200          # calibration samples (representative subset of the training)

os.makedirs(OUT, exist_ok=True)
imgs = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
# uniform sampling to vary scenes/light (not just the first N)
if len(imgs) > N:
    step = len(imgs) / N
    imgs = [imgs[int(i * step)] for i in range(N)]

lines = []
for i, path in enumerate(imgs):
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    canvas, _, _, _ = _letterbox(bgr, IMG_SIZE)          # letterbox 320x320
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)   # CHW
    arr = chw[np.newaxis, ...].copy()                    # (1,3,320,320) NCHW
    raw = os.path.join(OUT, f"calib_{i:04d}.raw")
    arr.astype(np.float32).tofile(raw)
    lines.append(raw)

# input_list: the quantizer accepts "<input_name>:=<path>" or just the path.
# Since there is 1 single input ("images"), the plain path is enough.
with open(os.path.join(OUT, "input_list.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"generated {len(lines)} .raw files in {OUT}")
print(f"input_list.txt with {len(lines)} lines")
print(f"shape per file: (1,3,{IMG_SIZE},{IMG_SIZE}) float32 = {1*3*IMG_SIZE*IMG_SIZE*4} bytes")
