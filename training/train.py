#!/usr/bin/env python3
"""
Training of PaddleDetNet (Phase A3).

Runs on the Mac: uses MPS (Metal GPU) if available, otherwise CPU.

Three concepts this file materializes:

1) COMBINED LOSS — the error has two parts:
     obj_loss = BCEWithLogits(obj)     -> "did it get right whether there IS a paddle?"
     box_loss = SmoothL1(cx,cy,w,h)    -> "is the box in the right place?"
   The box_loss is only charged on images that ACTUALLY have a paddle (obj=1).
   Charging box position on a background image makes no sense — there is no box.
   Total = obj_loss + LAMBDA_BOX * box_loss.

2) IoU (Intersection over Union) — the box quality metric:
   overlap area / union area. 0 = no touch, 1 = perfect box.
   It is the honest detection metric (the SmoothL1 loss is just the "fuel" of the
   gradient; the IoU is what actually matters to us).

3) OVERFITTING — with few images the network can memorize. We set aside a fraction
   of 'training' as VALIDATION (the 'testing' stays untouched for A4). If the
   training loss drops but the validation loss rises, it is a sign of memorization.

Usage:
    training/.venv/bin/python training/train.py --epochs 40
    training/.venv/bin/python training/train.py --epochs 40 --batch 32 --lr 1e-3
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import PaddleDataset
from model import PaddleDetNet

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_THIS_DIR, "checkpoints")

LAMBDA_BOX = 5.0        # box loss weight (position usually needs a push)
OBJ_THRESH = 0.5        # probability threshold to decide "has paddle"


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def boxes_iou(pred_box, true_box):
    """
    IoU between boxes in the [cx, cy, w, h] normalized format (tensors (N,4)).
    Converts to corners (x0,y0,x1,y1), computes intersection/union. Returns (N,).
    """
    def to_corners(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    p = to_corners(pred_box)
    t = to_corners(true_box)
    x0 = torch.max(p[:, 0], t[:, 0])
    y0 = torch.max(p[:, 1], t[:, 1])
    x1 = torch.min(p[:, 2], t[:, 2])
    y1 = torch.min(p[:, 3], t[:, 3])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_t = (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + 1e-9
    return inter / union


def run_epoch(model, loader, device, bce, optimizer=None):
    """One pass over the loader. If optimizer != None, trains; otherwise, evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss = total_obj = total_box = 0.0
    n = 0
    ious, obj_correct, n_pos = [], 0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        obj_true = y[:, 0:1]          # (B,1)
        box_true = y[:, 1:5]          # (B,4)

        with torch.set_grad_enabled(train):
            out = model(x)
            obj_logit = out[:, 0:1]
            box_pred = out[:, 1:5]

            obj_loss = bce(obj_logit, obj_true)
            # mask: only images with a paddle contribute to the box_loss
            mask = (obj_true[:, 0] > 0.5)
            if mask.any():
                box_loss = nn.functional.smooth_l1_loss(
                    box_pred[mask], box_true[mask]
                )
            else:
                box_loss = torch.zeros((), device=device)
            loss = obj_loss + LAMBDA_BOX * box_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_obj += obj_loss.item() * bs
        total_box += float(box_loss.detach()) * bs
        n += bs

        # metrics (no gradient)
        with torch.no_grad():
            prob = torch.sigmoid(obj_logit[:, 0])
            pred_has = prob > OBJ_THRESH
            obj_correct += (pred_has == (obj_true[:, 0] > 0.5)).sum().item()
            if mask.any():
                iou = boxes_iou(box_pred[mask], box_true[mask])
                ious.append(iou.cpu())
                n_pos += int(mask.sum())

    mean_iou = float(torch.cat(ious).mean()) if ious else 0.0
    return {
        "loss": total_loss / n,
        "obj_loss": total_obj / n,
        "box_loss": total_box / n,
        "obj_acc": obj_correct / n,
        "iou": mean_iou,
        "n_pos": n_pos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=CKPT_DIR)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    # the whole 'training' -> train/val split (the 'testing' stays for A4).
    full = PaddleDataset("training")
    n_val = int(len(full) * args.val_frac)
    n_tr = len(full) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full, [n_tr, n_val], generator=g)
    print(f"training: {n_tr}  |  validation: {n_val}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = PaddleDetNet().to(device)
    pos_weight = full.pos_weight().to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best_iou, best_state, history = -1.0, None, []

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, device, bce, optimizer)
        va = run_epoch(model, val_dl, device, bce)
        history.append({"epoch": ep, "train": tr, "val": va})
        print(
            f"ep {ep:02d}  "
            f"tr_loss {tr['loss']:.3f} (obj {tr['obj_loss']:.3f} box {tr['box_loss']:.3f})  "
            f"| val_loss {va['loss']:.3f}  val_objacc {va['obj_acc']:.3f}  "
            f"val_IoU {va['iou']:.3f}"
        )
        # we keep the best by the metric that matters: validation IoU
        if va["iou"] > best_iou:
            best_iou = va["iou"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    ckpt = os.path.join(args.out, "best.pt")
    torch.save({"model": best_state, "val_iou": best_iou, "img_size": 320}, ckpt)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"history": history, "best_val_iou": best_iou}, f, indent=2)
    print(f"\nbest val_IoU: {best_iou:.3f}")
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
