#!/usr/bin/env python3
"""Decodes the RAW output of the NPU (output0.raw) and checks whether it found the paddle.

The NPU ran the SAME graph as the .onnx and returned the tensor (1,5,2100) already
dequantized to float32 (qnn-net-run dequantizes the UFIXED_POINT_8 output using the
graph scales). Here we apply the SAME decode+NMS as web/infer_yolo.py — the only
difference is the origin of the numbers: NPU (Hexagon V75) instead of onnxruntime CPU.

Goal: prove that the NPU detects the paddle in emeet2.jpg (the frame that killed the CNN)
and compare the box/prob with the CPU/ONNX result.
"""
import os
import sys

import numpy as np


def _nms(boxes, scores, iou_thresh):
    """Classic NMS in numpy (copy of web/infer_yolo.py, no onnxruntime dep)."""
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x1 - x0) * (y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        w = np.maximum(0.0, xx1 - xx0)
        h = np.maximum(0.0, yy1 - yy0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


_THIS = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_THIS, "npu_out.raw")
META = os.path.join(_THIS, "emeet2_meta.txt")
OBJ_THRESH = 0.25
IOU_THRESH = 0.45

# --- read letterbox params (to undo them and return to the original pixels) ---
with open(META) as f:
    w0, h0, scale, padx, pady = f.read().split()
w0, h0 = int(w0), int(h0)
scale, padx, pady = float(scale), float(padx), float(pady)

# --- read the raw NPU output: (1,5,2100) float32 ---
out = np.fromfile(RAW, dtype=np.float32)
print(f"floats read: {out.size}  (expected 1*5*2100 = {5*2100})")
out = out.reshape(1, 5, 2100)

pred = out[0].T                    # (2100, 5): cx,cy,w,h,score
scores = pred[:, 4]
print(f"raw max score: {scores.max():.4f}  |  above {OBJ_THRESH}: {(scores>=OBJ_THRESH).sum()}")

mask = scores >= OBJ_THRESH
pred, scores = pred[mask], scores[mask]

if len(pred) == 0:
    print("\n>>> NO detection above the threshold. NPU did not find a paddle.")
    sys.exit(0)

cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
x0 = (cx - w / 2 - padx) / scale
y0 = (cy - h / 2 - pady) / scale
x1 = (cx + w / 2 - padx) / scale
y1 = (cy + h / 2 - pady) / scale
boxes = np.stack([x0, y0, x1, y1], axis=1)

keep = _nms(boxes, scores, IOU_THRESH)
boxes, scores = boxes[keep], scores[keep]
boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

print(f"\n>>> NPU detected {len(boxes)} paddle(s):")
for (bx0, by0, bx1, by1), sc in zip(boxes, scores):
    print(f"    box=({int(bx0)},{int(by0)})-({int(bx1)},{int(by1)})  prob={sc:.4f}")

print("\nCPU/ONNX (reference): box ~(224,177)-(383,363) prob ~0.74")
