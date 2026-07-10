#!/usr/bin/env python3
"""
Paddle-detection inference engine (Phase A5) — reusable on the Mac AND on the board.

The PaddleDetector class encapsulates everything that separates a "raw camera frame"
from a "box in image pixels": open the .onnx network, pre-process the frame EXACTLY
as in training, run it, and translate the normalized output back into pixels.

** Why this file is the same on the Mac and on the board **
onnxruntime reads the .onnx and picks a "provider" (the engine that runs the math):
  - CPUExecutionProvider — works everywhere (Mac and board). It's the baseline.
  - (later) a Qualcomm provider for the NPU — only changes the provider list here.
Since the pre/post-processing is identical, what we validate on the Mac is literally
what runs on the board. Only the provider and the camera index change.

** The pre-processing contract (must match preprocess.py) **
  BGR (OpenCV) -> RGB -> resize 320x320 (INTER_AREA) -> /255 -> CHW -> (1,3,320,320)
Any divergence here (channel order, scale) makes the network see something different
from what it trained on and the box comes out wrong.

** Network output **
  detection = [obj_logit, cx, cy, w, h]
  - obj_logit is RAW -> we apply sigmoid here to turn it into a probability 0..1.
  - cx,cy,w,h already come in 0..1 (sigmoid in the model) -> we multiply by the width/height
    of the ORIGINAL image to get pixels.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 320
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ONNX = os.path.join(_THIS_DIR, "..", "training", "checkpoints", "best.onnx")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Detection:
    """Result of one frame: is there a paddle? where? with what confidence?"""

    __slots__ = ("has_paddle", "prob", "x0", "y0", "x1", "y1")

    def __init__(self, has_paddle, prob, x0, y0, x1, y1):
        self.has_paddle = has_paddle    # bool: did prob pass the threshold?
        self.prob = prob                # float 0..1: confidence of "has paddle"
        self.x0, self.y0 = x0, y0       # top-left corner (pixels of the original image)
        self.x1, self.y1 = x1, y1       # bottom-right corner

    def __repr__(self):
        return (f"Detection(has={self.has_paddle} p={self.prob:.2f} "
                f"box=({self.x0},{self.y0})-({self.x1},{self.y1}))")


class PaddleDetector:
    def __init__(self, onnx_path=None, providers=None, obj_thresh=0.5):
        self.onnx_path = onnx_path or os.path.abspath(_DEFAULT_ONNX)
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(
                f".onnx model not found: {self.onnx_path} — run export_onnx.py first"
            )
        self.obj_thresh = obj_thresh

        opts = ort.SessionOptions()
        opts.log_severity_level = 3      # silence provider warnings (e.g.: GPU absent)
        # CPU is the baseline that runs on the Mac and on the board; the acceleration phase swaps this.
        self.providers = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=self.providers)
        self.input_name = self.sess.get_inputs()[0].name    # "image"
        self.output_name = self.sess.get_outputs()[0].name   # "detection"

    def _preprocess(self, frame_bgr):
        """BGR frame (H,W,3) uint8 -> tensor (1,3,320,320) float32, same as training."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        chw = resized.astype(np.float32) / 255.0
        chw = np.transpose(chw, (2, 0, 1))               # HWC -> CHW
        return chw[np.newaxis, ...]                      # -> (1,3,320,320)

    def detect(self, frame_bgr):
        """Run the network on a BGR frame and return a Detection in original-image pixels."""
        h0, w0 = frame_bgr.shape[:2]
        inp = self._preprocess(frame_bgr)
        out = self.sess.run([self.output_name], {self.input_name: inp})[0][0]  # (5,)

        obj_logit, cx, cy, w, h = out
        prob = float(_sigmoid(obj_logit))
        has = prob > self.obj_thresh

        # fractions 0..1 -> pixels of the ORIGINAL image (the box tracks the real frame)
        x0 = int((cx - w / 2) * w0)
        y0 = int((cy - h / 2) * h0)
        x1 = int((cx + w / 2) * w0)
        y1 = int((cy + h / 2) * h0)
        # clamp so we don't draw outside the frame
        x0, x1 = max(0, x0), min(w0, x1)
        y0, y1 = max(0, y0), min(h0, y1)
        return Detection(has, prob, x0, y0, x1, y1)

    def draw(self, frame_bgr, det):
        """Draw the box + label on the frame (in-place) when a paddle is detected."""
        if det.has_paddle:
            cv2.rectangle(frame_bgr, (det.x0, det.y0), (det.x1, det.y1), (0, 255, 0), 2)
            label = f"paddle {det.prob:.2f}"
            cv2.putText(frame_bgr, label, (det.x0, max(0, det.y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame_bgr


if __name__ == "__main__":
    # Smoke-test: run the detector on a test image from the cache and print the box.
    # (Validates that the .onnx loads and post-processing works, without needing a camera.)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="path to an image to test; if omitted uses the cache")
    args = ap.parse_args()

    det = PaddleDetector()
    print(f"model: {det.onnx_path}")
    print(f"active providers: {det.sess.get_providers()}")

    if args.image:
        frame = cv2.imread(args.image)
    else:
        # grab the 1st positive image from the test cache
        cache = os.path.join(_THIS_DIR, "..", "training", "cache")
        images = np.load(os.path.join(cache, "testing_images.npy"), mmap_mode="r")
        targets = np.load(os.path.join(cache, "testing_targets.npy"))
        idx = int(np.where(targets[:, 0] > 0.5)[0][0])
        rgb = np.ascontiguousarray(images[idx])
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        print(f"using test sample idx={idx}")

    result = det.detect(frame)
    print(result)
