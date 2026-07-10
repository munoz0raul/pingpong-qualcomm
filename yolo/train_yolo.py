#!/usr/bin/env python3
"""
Fine-tuning of YOLOv8-nano (Phase B2) — training on the Mac with MPS.

** Transfer learning, not training from scratch **
We start from `yolov8n.pt` (weights pre-trained on COCO: millions of images,
people/faces at every scale and lighting). The fine-tuning only needs to teach
"this here is a paddle" — the robustness to a large face / scale / light already
comes out of the box. It is the opposite of the Phase A CNN, trained from scratch
with ~300 positives, which memorized the training domain and broke in the real
environment (large face in close-up, dark room).

** imgsz=320 **
Aligns with Phase A and with what runs on the board. Ultralytics does letterbox
(keeps proportion with gray borders), so any camera resolution becomes 320x320
without distorting.

** free augmentation **
Ultralytics already applies mosaic, HSV (brightness/color), flip, etc. by default
— it attacks precisely light/framing, which is what took the CNN down.

** device=mps **
Metal on the Mac GPU. Training is on the Mac (project rule); inference later goes
to the board via ONNX (CPU -> NPU).

Usage:
    yolo/.venv/bin/python yolo/train_yolo.py
    yolo/.venv/bin/python yolo/train_yolo.py --epochs 120 --batch 32
"""

import argparse
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(_THIS_DIR, "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt",
                    help="base weights (downloaded automatically on the 1st run)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20,
                    help="early-stop: stops if val does not improve for N epochs")
    ap.add_argument("--device", default="mps", help="mps | cpu | 0")
    args = ap.parse_args()

    # import here so --help is instant (ultralytics is heavy to import)
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=os.path.join(_THIS_DIR, "runs"),
        name="paddle",
        exist_ok=True,          # overwrites runs/paddle instead of creating paddle2, paddle3...
        plots=True,             # generates loss/mAP curves + confusion matrix in runs/paddle/
    )

    best = os.path.join(_THIS_DIR, "runs", "paddle", "weights", "best.pt")
    print(f"\nbest checkpoint -> {best}")


if __name__ == "__main__":
    main()
