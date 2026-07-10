#!/usr/bin/env python3
"""
NPU inference engine (Phase D) — talks to the C++ daemon resident on the Hexagon V75.

Mirrors the PaddleDetector interface of Phases A/B (`.detect()` / `.draw()`), so
web/server.py switches to the NPU with `--model npu` without touching the MJPEG loop.

** Why a daemon, and not onnxruntime? **
The board does NOT have onnxruntime with the QNN EP nor a Python binding for QNN. The only path to
the NPU is the native C++ runtime (adapted SampleApp). But spinning up the process + loading the
context from the .bin costs ~250ms — unviable per frame. Solution: a C++ daemon that loads the
context ONCE and stays alive, running 1 inference per command. Here the Python:
    1. starts the daemon once (subprocess), waits for "DAEMON_READY"
    2. per frame: writes the input .raw -> signals "g" on cmd_fifo
                  -> waits for the response on resp_fifo -> reads output0.raw -> decode+NMS
The pure graphExecute is ~1.7ms; the rest is file I/O (fast, all in /tmp/tmpfs).

** Pre-processing and decode: IDENTICAL to infer_yolo.py **
Same letterbox (320, gray 114), same /255+CHW, same output (1,5,2100) and same
decode+NMS. We reuse the infer_yolo functions to guarantee that the box matches 1:1 with
what the model produces on CPU. The ONLY difference is where the matrix multiply runs.
"""

import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# reuse EVERYTHING from the YOLO post-processing (letterbox, nms, Detection, decode)
from infer_yolo import _letterbox, _nms, Detection, IMG_SIZE

# --- board defaults (IQ8-275) ---
NPU_DIR = "/home/weston/npu"                         # where daemon + .bin + libs live
DAEMON_BIN = "qnn-daemon-aarch64"
CONTEXT_BIN = "best_a16w8_htpv75.bin"
BACKEND = "libQnnHtp.so"
SYSTEM_LIB = "libQnnSystem.so"

CMD_FIFO = "/tmp/npu_cmd.fifo"
RESP_FIFO = "/tmp/npu_resp.fifo"
IN_FILE = "/tmp/npu_in.raw"                           # frame NCHW float32 (1,3,320,320)
OUT_DIR = "/tmp/npu_out"                              # daemon writes Result_0/output0.raw
OUT_RAW = os.path.join(OUT_DIR, "Result_0", "output0.raw")

N_CHANNELS = 5           # nc=1 -> 4 (box) + 1 (score)
N_ANCHORS = 2100         # for imgsz=320 (the A16W8 model generates 2100, not 8400)


class PaddleDetector:
    """Detector via NPU. Interface identical to Phases A/B; internally talks to the daemon."""

    def __init__(self, onnx_path=None, obj_thresh=0.25, iou_thresh=0.45,
                 npu_dir=NPU_DIR, context_bin=CONTEXT_BIN):
        self.obj_thresh = obj_thresh
        self.iou_thresh = iou_thresh
        self.npu_dir = npu_dir
        self.onnx_path = os.path.join(npu_dir, context_bin)   # only for logging; it's the .bin, not .onnx
        self.proc = None

        self._setup_fifos()
        self._start_daemon(context_bin)
        # cmd_fifo kept open for writing the whole session: each frame is just write+flush,
        # without reopening (reopening would trigger EOF in the daemon's getline).
        self._cmd = open(CMD_FIFO, "w")

    # ------------------------------------------------------------------ setup
    def _setup_fifos(self):
        for f in (CMD_FIFO, RESP_FIFO):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
            os.mkfifo(f)
        os.makedirs(os.path.join(OUT_DIR, "Result_0"), exist_ok=True)
        # input_list points to IN_FILE (the daemon re-reads this file every frame)
        self._il = "/tmp/npu_il.txt"
        with open(self._il, "w") as f:
            f.write(IN_FILE + "\n")
        # write a seed frame (zeros) so the 1st populate doesn't find an empty file
        np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32).tofile(IN_FILE)

    def _start_daemon(self, context_bin):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = self.npu_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = [
            os.path.join(self.npu_dir, DAEMON_BIN),
            "--backend", BACKEND,
            "--retrieve_context", context_bin,
            "--system_library", SYSTEM_LIB,
            "--input_list", self._il,
            "--output_dir", OUT_DIR,
            "--daemon",
            "--cmd_fifo", CMD_FIFO,
            "--resp_fifo", RESP_FIFO,
            "--in_file", IN_FILE,
            "--out_file", OUT_RAW,
        ]
        # cwd = npu_dir to resolve libQnnHtp.so etc. by relative path
        self.proc = subprocess.Popen(
            cmd, cwd=self.npu_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        # wait for "DAEMON_READY" (context loaded). Generous timeout: init ~a few s.
        t0 = time.time()
        while time.time() - t0 < 60:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"daemon died at startup:\n{out[-2000:]}")
            line = self.proc.stdout.readline()
            if not line:
                continue
            if "DAEMON_READY" in line:
                # drain the rest of stdout in the background so the buffer doesn't fill and stall
                threading_drain(self.proc.stdout)
                return
        raise RuntimeError("daemon did not signal DAEMON_READY within 60s")

    # -------------------------------------------------------------- inference
    def _preprocess(self, frame_bgr):
        canvas, scale, padx, pady = _letterbox(frame_bgr, IMG_SIZE)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...].copy()
        return chw, scale, padx, pady

    def _run_npu(self, chw):
        """Write the frame, signal the daemon, wait for the response, read the raw output."""
        chw.astype(np.float32).tofile(IN_FILE)          # frame -> file
        self._cmd.write("g\n")                          # trigger 1 inference
        self._cmd.flush()
        with open(RESP_FIFO, "r") as r:                 # block until the daemon responds
            ans = r.readline().strip()
        if ans != "1":
            raise RuntimeError(f"daemon reported an inference failure (resp={ans!r})")
        out = np.fromfile(OUT_RAW, dtype=np.float32)
        return out.reshape(1, N_CHANNELS, N_ANCHORS)

    def detect(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]
        chw, scale, padx, pady = self._preprocess(frame_bgr)
        out = self._run_npu(chw)                        # (1, 5, 2100) — same shape as YOLO

        pred = out[0].T                                 # (2100, 5): cx,cy,w,h,score
        scores = pred[:, 4]
        mask = scores >= self.obj_thresh
        pred, scores = pred[mask], scores[mask]
        if len(pred) == 0:
            return Detection(False, 0.0, 0, 0, 0, 0, [])

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x0 = (cx - w / 2 - padx) / scale
        y0 = (cy - h / 2 - pady) / scale
        x1 = (cx + w / 2 - padx) / scale
        y1 = (cy + h / 2 - pady) / scale
        boxes = np.stack([x0, y0, x1, y1], axis=1)

        keep = _nms(boxes, scores, self.iou_thresh)
        boxes, scores = boxes[keep], scores[keep]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

        det_boxes = [(int(a), int(b), int(c), int(d), float(s))
                     for (a, b, c, d), s in zip(boxes, scores)]
        bx0, by0, bx1, by1, best = det_boxes[0]
        return Detection(True, best, bx0, by0, bx1, by1, det_boxes)

    def draw(self, frame_bgr, det):
        for (x0, y0, x1, y1, prob) in det.boxes:
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"paddle {prob:.2f}", (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame_bgr

    def close(self):
        try:
            self._cmd.write("q\n")                      # ask the daemon to shut down
            self._cmd.flush()
            self._cmd.close()
        except Exception:
            pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    # compat with server.py: exposes .sess.get_providers() in the startup log
    class _FakeSess:
        def get_providers(self):
            return ["QnnHtp (NPU) via daemon C++"]

    @property
    def sess(self):
        return PaddleDetector._FakeSess()


def threading_drain(stream):
    """Consume the daemon's stdout in the background (avoids stalling on a full buffer)."""
    import threading

    def _drain():
        for _ in iter(stream.readline, ""):
            pass

    t = threading.Thread(target=_drain, daemon=True)
    t.start()


if __name__ == "__main__":
    # Smoke-test on the board: run 1 image through the NPU and print the box.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/home/weston/npu/emeet2.jpg")
    args = ap.parse_args()

    det = PaddleDetector()
    print(f"NPU context: {det.onnx_path}")
    print(f"providers: {det.sess.get_providers()}")

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"image not found: {args.image}")
        sys.exit(1)
    t0 = time.time()
    result = det.detect(frame)
    dt = (time.time() - t0) * 1000
    print(f"{result}   ({dt:.1f}ms round-trip)")
    if result.boxes:
        det.draw(frame, result)
        cv2.imwrite("/tmp/npu_infer_test.jpg", frame)
        print("annotated -> /tmp/npu_infer_test.jpg")
    det.close()
