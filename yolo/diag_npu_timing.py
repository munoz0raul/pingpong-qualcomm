#!/usr/bin/env python3
"""Diagnostic: where the time goes in the NPU cycle (write raw / FIFO / execute / read out / decode)."""
import os, sys, time
import numpy as np
import cv2
sys.path.insert(0, "/opt/pingpong/web")
from infer_npu import PaddleDetector, IN_FILE, RESP_FIFO, OUT_RAW, N_CHANNELS, N_ANCHORS
from infer_yolo import _letterbox, IMG_SIZE

det = PaddleDetector()
frame = cv2.imread("/tmp/emeet2.jpg")

# pre-process 1x (fixed)
canvas, scale, padx, pady = _letterbox(frame, IMG_SIZE)
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
chw = (rgb.astype(np.float32)/255.0).transpose(2,0,1)[np.newaxis,...].copy()

N = 50
acc = {"write":0.0, "signal":0.0, "wait":0.0, "read":0.0}
# warmup
for _ in range(5):
    det._run_npu(chw)

for _ in range(N):
    t0 = time.perf_counter()
    chw.astype(np.float32).tofile(IN_FILE)
    t1 = time.perf_counter()
    det._cmd.write("g\n"); det._cmd.flush()
    t2 = time.perf_counter()
    with open(RESP_FIFO,"r") as r:
        ans = r.readline().strip()
    t3 = time.perf_counter()
    out = np.fromfile(OUT_RAW, dtype=np.float32).reshape(1,N_CHANNELS,N_ANCHORS)
    t4 = time.perf_counter()
    acc["write"]  += (t1-t0)*1000
    acc["signal"] += (t2-t1)*1000
    acc["wait"]   += (t3-t2)*1000   # FIFO out + execute NPU + write out by the daemon + FIFO back
    acc["read"]   += (t4-t3)*1000

print(f"averages over {N} iters (ms):")
for k,v in acc.items():
    print(f"  {k:8s}: {v/N:.3f}")
print(f"  TOTAL   : {sum(acc.values())/N:.3f}")
det.close()
