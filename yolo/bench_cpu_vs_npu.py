#!/usr/bin/env python3
"""
Benchmark CPU vs NPU on the board — same pipeline, same frame, measured per stage.

Runs the SAME frame (emeet2.jpg) through the two Phase D engines:
  - CPU: infer_yolo.py  (onnxruntime CPUExecutionProvider)
  - NPU: infer_npu.py   (C++ daemon on the Hexagon V75)
Both expose .detect(); we measure the entire .detect() call (pre + infer + post),
which is what the MJPEG stream actually pays per frame. Warmup + N iterations.
"""
import os
import sys
import time
import statistics as st

import cv2

sys.path.insert(0, "/opt/pingpong/web")

IMG = "/tmp/emeet2.jpg"
WARMUP = 5
ITERS = 50


def bench(name, det, frame):
    # warmup (warms the NPU cache / onnxruntime JIT)
    for _ in range(WARMUP):
        det.detect(frame)
    times = []
    last = None
    for _ in range(ITERS):
        t0 = time.perf_counter()
        last = det.detect(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mean = st.mean(times)
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    fps = 1000.0 / mean
    print(f"\n[{name}]")
    print(f"  detection: {last}")
    print(f"  min={times[0]:.2f}ms  p50={p50:.2f}ms  mean={mean:.2f}ms  "
          f"p90={p90:.2f}ms  max={times[-1]:.2f}ms")
    print(f"  throughput: {fps:.1f} FPS  (over full .detect())")
    return mean, fps, last


def main():
    frame = cv2.imread(IMG)
    if frame is None:
        print(f"ERROR: could not open {IMG}")
        sys.exit(1)
    print(f"frame: {IMG}  shape={frame.shape}  |  warmup={WARMUP} iters={ITERS}")

    results = {}

    # --- CPU ---
    from infer_yolo import PaddleDetector as CpuDet
    cpu = CpuDet()
    results["CPU (onnxruntime)"] = bench("CPU (onnxruntime)", cpu, frame)

    # --- NPU ---
    from infer_npu import PaddleDetector as NpuDet
    npu = NpuDet()
    results["NPU (Hexagon V75)"] = bench("NPU (Hexagon V75)", npu, frame)
    npu.close()

    # --- comparative summary ---
    cpu_mean, cpu_fps, cpu_det = results["CPU (onnxruntime)"]
    npu_mean, npu_fps, npu_det = results["NPU (Hexagon V75)"]
    print("\n" + "=" * 56)
    print("SUMMARY — full .detect() (pre + inference + post)")
    print("=" * 56)
    print(f"  CPU:  {cpu_mean:7.2f} ms/frame   {cpu_fps:6.1f} FPS")
    print(f"  NPU:  {npu_mean:7.2f} ms/frame   {npu_fps:6.1f} FPS")
    print(f"  speedup (latency): {cpu_mean / npu_mean:.1f}x")
    print(f"  throughput gain: {npu_fps / cpu_fps:.1f}x")
    print(f"\n  CPU box: ({cpu_det.x0},{cpu_det.y0})-({cpu_det.x1},{cpu_det.y1}) p={cpu_det.prob:.4f}")
    print(f"  NPU box: ({npu_det.x0},{npu_det.y0})-({npu_det.x1},{npu_det.y1}) p={npu_det.prob:.4f}")


if __name__ == "__main__":
    main()
