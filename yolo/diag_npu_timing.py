#!/usr/bin/env python3
"""Diagnostic: where the time goes in the in-memory NPU cycle
(quantize / write / signal / wait / read+dequantize)."""
import os, sys, time
import numpy as np
import cv2
sys.path.insert(0, "/opt/pingpong/web")
from infer_npu import PaddleDetector, IN_FILE, RESP_FIFO, OUT_RAW, N_CHANNELS, N_ANCHORS
from infer_yolo import _letterbox, IMG_SIZE

det = PaddleDetector()
q = det.quant
frame = cv2.imread("/tmp/emeet2.jpg")

# pre-process 1x (fixed)
canvas, scale, padx, pady = _letterbox(frame, IMG_SIZE)
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
chw = (rgb.astype(np.float32)/255.0).transpose(2,0,1)[np.newaxis,...].copy()

N = 50
acc = {"quant":0.0, "write":0.0, "signal":0.0, "wait":0.0, "read_deq":0.0}
# warmup
for _ in range(5):
    det._run_npu(chw)

for _ in range(N):
    t0 = time.perf_counter()
    qin = np.rint(chw.astype(np.float32).ravel() / q["in_scale"] - q["in_offset"])
    qin = np.clip(qin, 0, 65535).astype(np.uint16)
    t1 = time.perf_counter()
    qin.tofile(IN_FILE)
    t2 = time.perf_counter()
    det._cmd.write("r\n"); det._cmd.flush()
    t3 = time.perf_counter()
    with open(RESP_FIFO,"r") as r:
        ans = r.readline().strip()
    t4 = time.perf_counter()
    qout = np.fromfile(OUT_RAW, dtype=np.uint16)
    out = ((qout.astype(np.float32) + q["out_offset"]) * q["out_scale"]).reshape(1,N_CHANNELS,N_ANCHORS)
    t5 = time.perf_counter()
    acc["quant"]    += (t1-t0)*1000   # vectorized float32 -> uint16
    acc["write"]    += (t2-t1)*1000   # native bytes -> file (RAM tmpfs)
    acc["signal"]   += (t3-t2)*1000
    acc["wait"]     += (t4-t3)*1000   # FIFO out + daemon memcpy+execute+write + FIFO back
    acc["read_deq"] += (t5-t4)*1000   # read uint16 + vectorized dequantize

print(f"averages over {N} iters (ms):")
for k,v in acc.items():
    print(f"  {k:9s}: {v/N:.3f}")
print(f"  TOTAL    : {sum(acc.values())/N:.3f}")
det.close()
