#!/usr/bin/env python3
"""
Export of the trained YOLOv8: best.pt -> best.onnx (Phase B3).

Same idea as Phase A (export_onnx.py): we train on the Mac, but the board runs
inference with onnxruntime, which reads the .onnx format (portable graph, no need
for PyTorch). Export once and run anywhere (Mac, board CPU, later NPU).

** Difference vs. Phase A **
There the export was a manual torch.onnx.export. Here Ultralytics itself exports
via `model.export(format="onnx")` — it already embeds the correct pre/post-processing
into the graph and chooses the input/output names. We keep imgsz=320 (aligns with
the training and with the board) and opset 17 (supported by onnxruntime and by the
Qualcomm tools).

** nms=False (important decision for the board) **
We do NOT embed the NMS in the graph. The embedded NMS uses ops (NonMaxSuppression)
that not every backend/NPU accepts. We do the decode + NMS in numpy in infer_yolo.py
(Phase B4), following the "post-processing in code, the graph only does the
convolutions" philosophy. This sets the stage for the NPU later.

Usage:
    yolo/.venv/bin/python yolo/export_yolo.py
    yolo/.venv/bin/python yolo/export_yolo.py --ckpt yolo/runs/paddle/weights/best.pt
"""

import argparse
import os
import shutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(_THIS_DIR, "runs", "paddle", "weights", "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(_THIS_DIR, "best.onnx"),
                    help="final path of the .onnx (default: yolo/best.onnx)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"checkpoint not found: {args.ckpt} — run train_yolo.py first")

    from ultralytics import YOLO

    model = YOLO(args.ckpt)
    # model.export returns the path of the generated .onnx (sits next to best.pt)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        nms=False,          # decode+NMS in numpy in infer_yolo.py (see docstring)
        dynamic=False,      # fixed shape 1x3x320x320 (the NPU requires fixed later)
        simplify=True,      # simplifies the graph (smaller/faster)
    )

    # copy to a stable path (yolo/best.onnx) that infer_yolo.py uses by default
    onnx_path = str(onnx_path)
    if os.path.abspath(onnx_path) != os.path.abspath(args.out):
        shutil.copy2(onnx_path, args.out)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nexported: {args.out}  ({size_kb:.0f} KB)")
    print(f"opset {args.opset} | input (1,3,{args.imgsz},{args.imgsz}) | NMS: in numpy (infer_yolo.py)")


if __name__ == "__main__":
    main()
