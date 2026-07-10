#!/usr/bin/env python3
"""
YOLOv8 inference engine (Phase B4) — onnxruntime + decode/NMS in numpy.

Mirrors the PaddleDetector interface from Phase A (`.detect()` / `.draw()`), so that
web/server.py can SWITCH between the two models with a `--model yolo|cnn` flag
without touching the MJPEG loop. What changes from one to the other is ONLY the post-processing.

** Why decode+NMS in numpy (and not embedded in the .onnx) **
We export with nms=False (see export_yolo.py). The .onnx graph does only the convolutions
and returns raw tensors; the decode (turning the output into boxes) and the NMS (removing
duplicate boxes of the same object) run here in numpy. Reason: the embedded NonMaxSuppression
op is not always accepted by the NPU; by keeping post-processing in code, the
same .onnx runs on CPU today and on the NPU later. It's the same philosophy as Phase A.

** The YOLOv8 pre-processing contract (letterbox!) **
  BGR -> RGB -> LETTERBOX to 320x320 -> /255 -> CHW -> (1,3,320,320)
Crucial difference vs. the Phase A CNN: the CNN did a DIRECT resize (stretched the image).
YOLO uses LETTERBOX: it resizes keeping the aspect ratio and pads the borders with
gray (114). That way the paddle isn't distorted. We need to undo the letterbox in the decode
(subtract the padding and the scale) so the box returns to the original-image pixels.

** YOLOv8 export output format **
  (1, 4+nc, N)  -> here nc=1, so (1, 5, 8400).
  Rows 0..3 = cx,cy,w,h in PIXELS of the 320x320 space (NOT normalized).
  Row   4    = score of the 'paddle' class (sigmoid already baked into the graph).
We transpose to (8400, 5), filter by score, convert to corners, NMS, and
undo the letterbox.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 320
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ONNX = os.path.join(_THIS_DIR, "..", "yolo", "best.onnx")


class Detection:
    """Same interface as Phase A: is there a paddle? where? with what confidence?

    Here `has_paddle` = "there is at least 1 box above the threshold"; the stored
    box/prob are those of the HIGHEST-confidence detection (the main paddle).
    To draw ALL boxes, use `boxes` (list of (x0,y0,x1,y1,prob)).
    """

    __slots__ = ("has_paddle", "prob", "x0", "y0", "x1", "y1", "boxes")

    def __init__(self, has_paddle, prob, x0, y0, x1, y1, boxes=None):
        self.has_paddle = has_paddle
        self.prob = prob
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.boxes = boxes or []

    def __repr__(self):
        return (f"Detection(has={self.has_paddle} p={self.prob:.2f} "
                f"n={len(self.boxes)} box=({self.x0},{self.y0})-({self.x1},{self.y1}))")


def _letterbox(img, new_size=IMG_SIZE, color=114):
    """Resize keeping aspect ratio + pad borders (same as YOLO training).

    Returns (320x320 image, scale, padx, pady) — the last 3 serve to undo
    the letterbox in the decode and recover the original-image pixels.
    """
    h0, w0 = img.shape[:2]
    scale = min(new_size / w0, new_size / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    padx, pady = (new_size - nw) // 2, (new_size - nh) // 2
    canvas[pady:pady + nh, padx:padx + nw] = resized
    return canvas, scale, padx, pady


def _nms(boxes, scores, iou_thresh):
    """Classic NMS in numpy: sort by score, remove heavily overlapping boxes."""
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


class PaddleDetector:
    """YOLOv8 detector via onnxruntime. Interface identical to Phase A's."""

    def __init__(self, onnx_path=None, providers=None, obj_thresh=0.25, iou_thresh=0.45):
        self.onnx_path = onnx_path or os.path.abspath(_DEFAULT_ONNX)
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(
                f".onnx model not found: {self.onnx_path} — run export_yolo.py first")
        self.obj_thresh = obj_thresh      # YOLO uses 0.25 by default
        self.iou_thresh = iou_thresh

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.providers = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=self.providers)
        self.input_name = self.sess.get_inputs()[0].name       # "images"
        self.output_name = self.sess.get_outputs()[0].name      # "output0"

    def _preprocess(self, frame_bgr):
        canvas, scale, padx, pady = _letterbox(frame_bgr, IMG_SIZE)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = rgb.astype(np.float32) / 255.0
        chw = np.transpose(chw, (2, 0, 1))
        return chw[np.newaxis, ...], scale, padx, pady

    def detect(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]
        inp, scale, padx, pady = self._preprocess(frame_bgr)
        out = self.sess.run([self.output_name], {self.input_name: inp})[0]  # (1, 5, 8400)

        pred = out[0].T                       # (8400, 5): cx,cy,w,h,score
        scores = pred[:, 4]
        mask = scores >= self.obj_thresh
        pred, scores = pred[mask], scores[mask]

        if len(pred) == 0:
            return Detection(False, 0.0, 0, 0, 0, 0, [])

        # cx,cy,w,h (320 letterboxed space) -> corners, then undo letterbox
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x0 = (cx - w / 2 - padx) / scale
        y0 = (cy - h / 2 - pady) / scale
        x1 = (cx + w / 2 - padx) / scale
        y1 = (cy + h / 2 - pady) / scale
        boxes = np.stack([x0, y0, x1, y1], axis=1)

        keep = _nms(boxes, scores, self.iou_thresh)
        boxes, scores = boxes[keep], scores[keep]

        # clamp to the original-image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

        det_boxes = []
        for (bx0, by0, bx1, by1), sc in zip(boxes, scores):
            det_boxes.append((int(bx0), int(by0), int(bx1), int(by1), float(sc)))

        # the main detection = highest confidence (NMS already sorted by score)
        bx0, by0, bx1, by1, best = det_boxes[0]
        return Detection(True, best, bx0, by0, bx1, by1, det_boxes)

    def draw(self, frame_bgr, det):
        """Draw ALL YOLO boxes (multi-object), green + confidence."""
        for (x0, y0, x1, y1, prob) in det.boxes:
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label = f"paddle {prob:.2f}"
            cv2.putText(frame_bgr, label, (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame_bgr


if __name__ == "__main__":
    # Smoke-test: run on an image and print the boxes (validates .onnx + decode/NMS).
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(_THIS_DIR, "..", "yolo", "test_frames", "emeet2.jpg"))
    ap.add_argument("--onnx", default=None)
    args = ap.parse_args()

    det = PaddleDetector(onnx_path=args.onnx)
    print(f"model: {det.onnx_path}")
    print(f"active providers: {det.sess.get_providers()}")

    frame = cv2.imread(args.image)
    result = det.detect(frame)
    print(result)
    if result.boxes:
        out = "/tmp/yolo_infer_test.jpg"
        det.draw(frame, result)
        cv2.imwrite(out, frame)
        print(f"annotated -> {out}")
