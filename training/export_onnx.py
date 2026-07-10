#!/usr/bin/env python3
"""
Export of the trained model: best.pt -> best.onnx (Phase A5).

Why ONNX?  The .pt is a PyTorch format — to run it you need the whole PyTorch
installed. The board (IQ8-275) runs inference with onnxruntime, which is light
and reads the .onnx format (a portable operation graph, framework-independent).
Exporting once on the Mac and running anywhere (Mac, board CPU, later NPU) is
exactly the idea: TRAIN in one place, INFER in another.

What the export does: takes the network in eval mode (turns off
Dropout/BatchNorm-training), passes a "fake" (dummy) tensor of the right shape
so PyTorch can "record" which operations the network runs, and serializes that
graph into the .onnx file.

opset 17: the "dictionary version" of ONNX operations. 17 is widely supported
by onnxruntime (Mac and board) and by the Qualcomm tools of the acceleration
phase.

Usage:
    training/.venv/bin/python training/export_onnx.py
    training/.venv/bin/python training/export_onnx.py --ckpt training/checkpoints/best.pt
"""

import argparse
import os

import torch

from model import PaddleDetNet, IMG_SIZE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_THIS_DIR, "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CKPT_DIR, "best.pt"))
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "best.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt} — run train.py first")

    # We ALWAYS export on CPU: the ONNX graph is device-independent, and it avoids
    # surprises with MPS types. The inference picks the provider later.
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = PaddleDetNet()
    model.load_state_dict(ckpt["model"])
    model.eval()   # CRITICAL: turns off Dropout and puts BatchNorm in inference mode

    # dummy input: 1 image, 3 channels, 320x320 — only the SHAPE matters, not the values.
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["image"],           # input name (used in infer.py)
        output_names=["detection"],      # output (1,5) = [obj_logit, cx, cy, w, h]
        opset_version=args.opset,
        do_constant_folding=True,        # pre-computes constants -> smaller/faster graph
        dynamic_axes=None,               # fixed shape 1x3x320x320 (the NPU requires fixed later)
    )

    val_iou = ckpt.get("val_iou", float("nan"))
    size_kb = os.path.getsize(args.out) / 1024
    print(f"exported: {args.out}  ({size_kb:.0f} KB)")
    print(f"checkpoint val_IoU: {val_iou:.3f}")
    print(f"opset {args.opset} | input 'image' (1,3,{IMG_SIZE},{IMG_SIZE}) | output 'detection' (1,5)")

    # sanity check: reload with onnxruntime and compare with PyTorch on the same input
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(["detection"], {"image": dummy.numpy()})[0]
        with torch.no_grad():
            torch_out = model(dummy).numpy()
        max_diff = float(np.abs(onnx_out - torch_out).max())
        print(f"ONNX vs PyTorch check: max difference = {max_diff:.2e}  "
              f"({'OK' if max_diff < 1e-4 else 'WARNING: divergence'})")
    except ImportError:
        print("(onnxruntime not installed — skipped the check)")


if __name__ == "__main__":
    main()
