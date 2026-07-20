#!/usr/bin/env python3
"""Generates the NCHW .raw of a test frame (same letterbox as inference) for qnn-net-run.

Usage: gen_test_input.py [IMAGE]
  IMAGE defaults to yolo/test_frames/emeet2.jpg (my hard frame, NOT shipped in the repo).
  Pass the hard frame you captured in Step 4.5, e.g.
      yolo/.venv/bin/python yolo/gen_test_input.py /path/to/your_hard_frame.jpg
"""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))
from infer_yolo import _letterbox, IMG_SIZE

img = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(__file__), "test_frames", "emeet2.jpg")
bgr = cv2.imread(img)
if bgr is None:
    sys.exit(f"ERROR: can't read image: {img}\n"
             f"       Pass the hard frame you used in Step 4.5:\n"
             f"       gen_test_input.py /path/to/your_hard_frame.jpg")
canvas, scale, padx, pady = _letterbox(bgr, IMG_SIZE)
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...].copy()

out = os.path.join(os.path.dirname(__file__), "emeet2_input.raw")
chw.astype(np.float32).tofile(out)
# save the letterbox params to decode the output later
meta = os.path.join(os.path.dirname(__file__), "emeet2_meta.txt")
h0, w0 = bgr.shape[:2]
with open(meta, "w") as f:
    f.write(f"{w0} {h0} {scale} {padx} {pady}\n")
print(f"input .raw: {out}  shape (1,3,{IMG_SIZE},{IMG_SIZE}) = {chw.nbytes} bytes")
print(f"letterbox: w0={w0} h0={h0} scale={scale:.5f} padx={padx} pady={pady}")
