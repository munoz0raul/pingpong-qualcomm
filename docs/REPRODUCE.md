# Reproduce it yourself, end to end

This is the full, no-steps-hidden guide, from an empty folder to live paddle detection on
a Qualcomm NPU. No prior AI experience assumed. Every command is here; every script it
calls is in this repo with English comments.

**Phases A and B (Steps 1–6) are copy-paste** on a Mac + the board. **Phases C and D
(Steps 7–9)** — the Qualcomm NPU part — need the (free) QAIRT SDK, an x86 Linux box, and a
little comfort editing paths in a shell script; they're a *worked reference* rather than
turnkey. Each of those steps says exactly what it assumes.

**What you'll go through:**

1. [Build a dataset](#step-1)
2. [Set up the Mac training environment](#step-2)
3. [Phase A — train a CNN from scratch](#step-3) *(the baseline that teaches the lesson)*
4. [Phase B — fine-tune YOLOv8n](#step-4) *(the one that actually works)*
5. [Run it live in a browser on the Mac](#step-5)
6. [Run it on the Qualcomm board's CPU](#step-6)
7. [Convert the model for the Qualcomm NPU](#step-7)
8. [Run it on the NPU — one-shot, then live via a C++ daemon](#step-8)
9. [Benchmark CPU vs NPU honestly](#step-9)

> **The golden rule of this project:** *train on the big computer, run on the little
> one.* All training happens on a Mac (PyTorch + Apple's MPS/Metal). All inference —
> first on CPU, then NPU — happens on the board.

---

## What you need

**Hardware**
- A Mac with Apple Silicon (M1/M2/M3…) for training. (Any machine with PyTorch works;
  the `--device mps` flag is Mac-specific — use `cpu` elsewhere.)
- A **Qualcomm IQ8-275 EVK** board (QCS8300, Hexagon **V75** NPU), or a similar Qualcomm
  board. It runs Linux, aarch64, Python 3.14, with `onnxruntime` preinstalled.
- A USB webcam for the live demo. (Mine was an EMEET SmartCam; on the board it showed up
  as `/dev/video26`.)
- An **x86-64 Linux machine** to run Qualcomm's conversion toolkit (the QAIRT SDK is
  x86-only). This can be any Linux box/VM.

**Software**
- On the Mac: Python 3, `git`.
- On the x86 Linux box: Qualcomm's **QAIRT SDK** (I used v2.47.0.260601), a free download
  from the Qualcomm developer site. It contains `qairt-converter`,
  `qairt-quantizer`, `qnn-context-binary-generator`, and `qnn-net-run`.
- To build the live-NPU C++ daemon: an aarch64 cross-compiler
  (`aarch64-linux-gnu-g++-13`). [Step 8](#step-8) covers this.

---

## Step 1 — Build a dataset<a name="step-1"></a>

The dataset is the foundation. You need images of the object (paddle) with a box drawn
around it in each, plus some background images with no paddle.

**The easy path (what I used): [Edge Impulse](https://edgeimpulse.com).** It's a free
web tool where you upload images/video, draw bounding boxes in the browser, and export.
Choose the **"Object Detection"** project type and export in the format that gives you,
per split, a folder of images plus a `bounding_boxes.labels` JSON file.

Aim for a few hundred labeled images with **variety**: different distances, angles,
rooms, and lighting. Include ~10–20% background frames with no paddle. Variety matters
more than raw count — Phase A fails precisely because its images weren't varied enough.

You'll end up with an export folder like this (this repo's code expects exactly this):

```
pingpong-export/
  training/
    <image>.jpg ...
    bounding_boxes.labels     # JSON: filename -> list of {label, x, y, width, height}
  testing/
    <image>.jpg ...
    bounding_boxes.labels
```

The label boxes are in **absolute pixels**, with `x,y` = the **top-left corner**.
Remember that — different tools use different conventions, and getting it wrong silently
ruins training. `training/labels.py` is the single place that reads this format; every
other script goes through it.

---

## Step 2 — Mac training environment<a name="step-2"></a>

```bash
git clone https://github.com/munoz0raul/pingpong-qualcomm.git
cd pingpong-qualcomm

# Put your Edge Impulse export here (the code looks for training/ and testing/ inside it):
#   pingpong-qualcomm/pingpong-export/training/...
#   pingpong-qualcomm/pingpong-export/testing/...
```

We use **two separate virtual environments** on purpose: Phase A (a lean PyTorch setup)
and Phase B (Ultralytics/YOLO, which pulls in a lot more). Keeping them apart means
Phase A stays reproducible even after you install the heavier YOLO stack.

```bash
# Phase A environment
python3 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements.txt
```

---

## Step 3 — Phase A: a CNN from scratch (the instructive baseline)<a name="step-3"></a>

This phase builds a small neural network with no pre-training and teaches it only on your
paddle photos. **It will underperform** — that's the point. It's the control group that
proves why Phase B is worth it.

The pipeline is four independent scripts: **preprocess → train → eval → export**.

### 3.1 Preprocess (decode images once, cache to disk)

Decoding JPEGs every training epoch is slow. `preprocess.py` decodes each image once,
crops/resizes it to 320×320, and caches the result as a `.npy` array. It also converts
each label box into the network's target format — `[present, cx, cy, w, h]`, where the
center `cx,cy` and size `w,h` are **normalized 0–1** (this normalization, in
`box_to_target()`, is reused everywhere later).

```bash
training/.venv/bin/python training/preprocess.py
```

It finds the export on its own (through `labels.py`, which knows the `pingpong-export/`
layout) and caches both splits. Add `--force` to rebuild an existing cache, or
`--splits training testing` to pick specific splits.

### 3.2 Train

```bash
training/.venv/bin/python training/train.py --epochs 20
```

Key design choices, all in `train.py` with comments explaining why:

- **`pick_device()`** uses Apple's **MPS** (Metal GPU) if present, else CPU.
- **Temporal split, not random:** the first 80% of frames are training, the last 20% are
  validation. Video frames next to each other are near-identical; a random split would
  leak almost-copies into validation and give a dishonestly high score.
- **Balanced accuracy + weighted loss** so the model can't cheat by always guessing the
  majority class.
- Prints `val_IoU` every epoch and saves the best model + `metrics.json` under
  `training/checkpoints/`.

### 3.3 Read the result

There is no separate eval script in Phase A — `train.py` evaluates on the validation
split every epoch and prints it live. Watch the `val_IoU` figure: it's the honest quality
measure for a box (Intersection-over-Union: how much the predicted box overlaps the true
one, 0=miss, 1=perfect). It tops out around **IoU 0.56**. The full history is saved to
`training/checkpoints/metrics.json`, and the best checkpoint to
`training/checkpoints/best.pt`.

### 3.4 Export to ONNX (the portable format for the board)

```bash
training/.venv/bin/python training/export_onnx.py \
    --ckpt training/checkpoints/best.pt --out training/checkpoints/best.onnx
```

**ONNX** is a framework-neutral file the board can run without PyTorch installed. Opset
17 — supported by both `onnxruntime` and the Qualcomm tools.

### 3.5 Watch it fail (the valuable part)

Point a webcam at yourself, lean in close in a dim room. It won't find the paddle.
Diagnose it like we did: crop your face out of the frame → it suddenly works; brighten
the frame → the score rises. **Conclusion: the tiny model memorized its training domain
(distant, well-lit, full-body scenes) and can't generalize to a big close-up face in low
light.** That's the motivation for Phase B.

---

## Step 4 — Phase B: fine-tune YOLOv8n (the one that works)<a name="step-4"></a>

Instead of building from zero, we start from **YOLOv8-nano**, pre-trained on the COCO
dataset (millions of images, people and faces at every scale and lighting), and
**fine-tune** it on our few hundred paddle photos. It inherits all that visual robustness
for free.

### 4.1 Environment

```bash
python3 -m venv yolo/.venv
yolo/.venv/bin/pip install -r yolo/requirements.txt   # ultralytics + onnx + onnxruntime
```

### 4.2 Convert labels to YOLO format

YOLO wants, per image `foo.jpg`, a text file `foo.txt` with one line per object:
`<class> <cx> <cy> <w> <h>`, all normalized 0–1 with the **center** (not the corner).
`prep_yolo.py` reads your Edge Impulse export through the same `labels.py` and writes it
out, plus a `data.yaml` describing the dataset.

```bash
yolo/.venv/bin/python yolo/prep_yolo.py
# sanity check: draws boxes back onto images so you can eyeball the conversion
yolo/.venv/bin/python yolo/prep_yolo.py --check
```

The Edge Impulse `training/`→YOLO `train`, `testing/`→YOLO `val` split is preserved (no
leakage: they were distinct splits to begin with). Background images get an **empty**
`.txt` — YOLO uses those as negatives.

### 4.3 Fine-tune

```bash
yolo/.venv/bin/python yolo/train_yolo.py --epochs 80
# (defaults: yolov8n.pt base, imgsz=320, batch=16, device=mps)
```

- **`imgsz=320`** matches Phase A and what we'll run on the board. Ultralytics letterboxes
  (pads to a square with gray borders) so any camera resolution fits without distortion.
- **Free augmentation:** Ultralytics automatically applies mosaic, HSV (brightness/color)
  jitter, and flips — which directly attacks the lighting/framing problem that killed
  Phase A.
- Result lands in `yolo/runs/paddle/weights/best.pt`, plus loss/mAP curves and a
  confusion matrix in `yolo/runs/paddle/`.

The score here is **mAP@0.5** (the standard detection metric). Phase B reaches
**mAP@0.5 ≈ 0.98** — versus IoU 0.56 for the from-scratch CNN.

![YOLOv8n training curves](img/yolo_training_curves.png)
*Loss falling and mAP climbing over 80 epochs — the model learning "paddle."*

### 4.4 Export to ONNX

```bash
yolo/.venv/bin/python yolo/export_yolo.py   # reads runs/paddle/weights/best.pt -> yolo/best.onnx
```

(It defaults to the `runs/paddle/weights/best.pt` that 4.3 just produced; pass
`--ckpt <path>` to export a different one.)

Important choice: **`nms=False`** — we do **not** bake the final NMS step into the graph.
NMS uses operations not every NPU backend supports. We do decode + NMS in plain numpy in
`web/infer_yolo.py` instead. Philosophy: *the graph does only the convolutions; postprocessing
lives in code.* This is what makes the NPU port clean later. Shape is fixed at
1×3×320×320 (the NPU requires fixed shapes).

### 4.5 Prove the generalization

Grab the exact frame that killed Phase A (big face, paddle close, dim room — mine is
`emeet2.jpg`) and run YOLO on it. It finds the paddle (box ≈ (224,177)-(383,363),
confidence ≈ 0.75). Thesis proven: the pre-trained model generalizes where the
from-scratch one couldn't.

![YOLO finds the paddle in the hard frame](img/yolo_emeet2.jpg)
*The close-up-face, dim-room frame that blinded the from-scratch CNN — YOLOv8n boxes the paddle confidently.*

---

## Step 5 — Live demo in a browser (on the Mac)<a name="step-5"></a>

`web/server.py` is a tiny web server (Python standard library + OpenCV, no framework). It
opens the webcam, runs each frame through a detector, draws the box, and streams the
result to your browser as **MJPEG** (a sequence of JPEGs the browser flips through like
video). It exposes a small API: list cameras, probe supported resolutions, start, stop,
stream.

```bash
# 'yolo' selects web/infer_yolo.py; 'cnn' would select the Phase A model
yolo/.venv/bin/python web/server.py --model yolo
# open http://localhost:8080 , pick a camera, hit "start"
```

The key architectural trick: **every engine exposes the same `PaddleDetector` interface**
(`.detect()` and `.draw()`). So `infer.py` (CNN), `infer_yolo.py` (YOLO/CPU), and
`infer_npu.py` (YOLO/NPU) are interchangeable — the MJPEG loop in `server.py` never
changes. This one decision makes Steps 6–8 painless.

> ⚠️ **Always press "stop" in the browser before closing.** If you don't, the camera
> device stays locked open and won't reopen. (We added a guard so `/api/cameras` reports
> the in-use camera instead of failing to re-probe a locked device — but stopping cleanly
> is the habit.)

---

## Step 6 — Run on the Qualcomm board's CPU<a name="step-6"></a>

Now move to the board. SSH in (my board: `192.168.15.86`, user/pass provided with the
EVK):

```bash
# from the Mac — copy the model and the web/ code to the board
scp yolo/best.onnx  root@192.168.15.86:/opt/pingpong/yolo/
scp web/*.py        root@192.168.15.86:/opt/pingpong/web/

ssh root@192.168.15.86
```

The board already has `onnxruntime` (Python 3.14, aarch64). Because inference is just
`onnxruntime` with the `CPUExecutionProvider`, and the pre/post-processing is byte-identical
to the Mac, **what you validated on the Mac is literally what runs here**. Only the camera
index and the capture backend change.

```bash
# on the board — v4l2 backend is required on Linux (the default GStreamer backend
# breaks when you change resolution after opening)
cd /opt/pingpong
python3 web/server.py --model yolo --cameras 26 --backend v4l2 --port 8080
# open http://192.168.15.86:8080 from any machine on the network
```

You'll get ~24 FPS end-to-end on the CPU. That's your **CPU baseline**.

> 🔌 **Real-world gremlin:** if the USB camera disappears (gone from `lsusb`, `/dev/video26`
> missing), a hot-replug won't bring it back — it resets the USB hub. **Reboot the board.**
> Hot-plug fails; reboot works.

---

## Step 7 — Convert the model for the Qualcomm NPU<a name="step-7"></a>

This all happens on the **x86-64 Linux machine** with the QAIRT SDK installed. The scripts
are in `npu/`. Copy `yolo/best.onnx` and your calibration data to that machine first.

> ⚠️ **Honesty note for Steps 7–8.** Unlike Phases A/B, these are *not* pure copy-paste.
> The `npu/*.sh` scripts contain **absolute paths from my machine** (e.g.
> `/local/mnt/workspace/qairt/...`) and assume the QAIRT SDK is installed and on your
> `PATH`. Treat them as a **worked reference**, not a turnkey script: read each one, then
> set `SDK=<your QAIRT install>` and adjust the paths to your box. The commands below show
> the essential call for each stage; the scripts wrap them with those machine-specific
> paths. You also need Qualcomm's runtime libraries — from the SDK, the ones that end up on
> the board are `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpNetRunExtensions.so`, and the
> matching HTP **V75** skel/stub libs (`libQnnHtpV75*.so`), all from
> `<SDK>/lib/aarch64-*/` and `<SDK>/lib/hexagon-v75/`.

The NPU can't run the float ONNX directly. Three transformations:

**7.1 — ONNX → DLC (float).** `qairt-converter` translates the graph into Qualcomm's
`.dlc` format, still in floating point. See `npu/convert_dlc.sh`:

```bash
qairt-converter --input_network best.onnx --output_path best_fp.dlc
```

**7.2 — Generate calibration data.** Quantization (next step) learns each tensor's value
range by running the model on **real images**, which must go through the *exact* same
preprocessing as inference. `yolo/gen_calib.py` produces ~200 `.raw` files (float32 NCHW,
letterboxed 320² — reusing `_letterbox` from `infer_yolo.py` so it's bit-identical) plus
an `input_list.txt`. Run it on the Mac, copy the `calib/` folder to the x86 box.

**7.3 — Quantize → the A16W8 trap.** This is the subtle part. `qairt-quantizer` converts
the float model to small integers so the NPU runs fast. See `npu/requant_a16w8.sh`:

```bash
qairt-quantizer \
  --input_dlc best_fp.dlc \
  --input_list calib/input_list.txt \
  --act_bitwidth 16 --weights_bitwidth 8 \
  --output_dlc best_a16w8.dlc
```

> **⚠️ The trap:** plain INT8 (8-bit everything) **crushes the confidence score to zero**.
> Why: in this model the box coordinates (values 0–580) and the score (0–1) share the same
> quantization scale. The huge coordinate range flattens the delicate little score into
> nothing. **Fix:** use **16-bit activations, 8-bit weights** (`--act_bitwidth 16`,
> called **A16W8**). The extra activation precision protects the score. After this fix the
> NPU reports a healthy confidence (~0.75) again. If your quantized model "detects but with
> score 0," this is almost certainly why.

**7.4 — Build the NPU context binary.** `qnn-context-binary-generator` compiles the
quantized DLC into a `.bin` pre-optimized for the **Hexagon V75** HTP. This needs a small
JSON config naming the graph and target arch (`"htp_arch": "v75"`). See
`npu/requant_a16w8.sh` (it does 7.3 and 7.4 together) and `npu/gen_ctx2.sh`:

```bash
qnn-context-binary-generator \
  --dlc_path best_a16w8.dlc \
  --backend libQnnHtp.so \
  --binary_file best_a16w8_htpv75 \
  --config_file backend_ext.json
# -> best_a16w8_htpv75.bin
```

Copy `best_a16w8_htpv75.bin` and the runtime `.so` libraries from the SDK to the board
under `/home/weston/npu/`.

---

## Step 8 — Run on the NPU<a name="step-8"></a>

### 8.1 One-shot sanity check

The board has **no** `onnxruntime` QNN provider and no Python QNN binding — the only path
to the NPU is Qualcomm's native C++ runtime (`qnn-net-run`). First confirm the NPU works
on a single frame. `yolo/gen_test_input.py` writes a test frame as `.raw`; `npu/run_npu16.sh`
runs it through the NPU; `yolo/decode_npu_out.py` decodes the raw output and confirms the
paddle was found.

```bash
# on the board
bash npu/run_npu16.sh              # writes out16/Result_0/output0.raw
python3 yolo/decode_npu_out.py     # -> "NPU detected 1 paddle(s): box=... prob=0.74"
```

The **raw NPU compute is ~1.7 ms** — about **84× faster** than the CPU's 145 ms for the
same math. (Measured via `qnn-net-run --profiling`.)

### 8.2 Live — the C++ daemon

Here's the problem for live video: spinning up `qnn-net-run` fresh each frame costs
~250 ms just to load the context — hopeless. The fix is a **resident daemon**: a C++
program that loads the NPU context **once**, stays alive, and runs **one inference per
command**.

We built it by adapting Qualcomm's SDK `SampleApp`. The sources are in `npu/daemon/`:
- `QnnSampleApp.cpp` — the added **`runDaemon()`** method: set up the tensors once, then
  loop reading a command FIFO. Two per-frame paths: on `"g"` (legacy) it re-reads the input
  file and lets the SDK convert float↔uint16 element-by-element; on `"r"` (the fast in-memory
  path, see Step 9) it **memcpy's the already-native `uint16` bytes** straight into the tensor
  buffer and writes the raw output back. Either way it executes and replies `"1"`. On `"q"`:
  quit. At startup it prints a `QUANT` banner (input/output `scale`, `offset`, byte sizes) so
  the Python side can quantize/dequantize identically in numpy.
- `main.cpp` — the added `--daemon --cmd_fifo --resp_fifo --in_file --out_file` flags.

**Protocol** (Python ↔ daemon, via files + FIFOs, all in `/tmp` which is a RAM disk):
Python quantizes the frame in numpy, writes the native `uint16` bytes to `/tmp/npu_in.raw`,
sends `"r\n"` on the command FIFO; the daemon runs and replies `"1"` on the response FIFO;
Python reads the raw `uint16` result from `/tmp/npu_out/.../output0.raw` and dequantizes it.
`web/infer_npu.py` implements the Python side behind the *same* `PaddleDetector` interface,
so `server.py` runs the NPU with `--model npu` and the MJPEG loop is unchanged.

**Cross-compiling the daemon** (on the x86 box, targeting aarch64 — see
`npu/build_daemon.sh` for the exact, working recipe). The daemon is Qualcomm's SampleApp
sources plus our `runDaemon()`; you compile them against the SDK headers with an aarch64
cross-compiler. The variables below (spelled out in full in `build_daemon.sh`):

- `$R` — the sysroot of your aarch64 cross toolchain (where `aarch64-linux-gnu-g++-13` and
  its libs live).
- `$INCLUDES` — `-Isrc ... -I$SDK/include/QNN` (the SampleApp source tree + the SDK's QNN
  headers).
- `$SRCS` — `src/main.cpp src/QnnSampleApp.cpp` plus the SampleApp support dirs
  (`Log/`, `PAL/`, `Utils/`, `WrapperUtils/`).

```bash
aarch64-linux-gnu-g++-13 -std=c++17 --sysroot="$R" \
  -fPIC -Wno-write-strings -fno-exceptions -fno-rtti -DQNN_API= \
  $INCLUDES $SRCS -o qnn-daemon-aarch64 \
  -ldl -static-libstdc++ -static-libgcc
```

Copy `qnn-daemon-aarch64` to `/home/weston/npu/` on the board alongside the `.bin` and
`.so` files. Then run the live server on the NPU:

```bash
# on the board — setsid detaches it from the SSH session (a plain '&' dies on logout)
setsid python3 web/server.py --model npu --cameras 26 --backend v4l2 --port 8080 \
  > /tmp/srv_npu.log 2>&1 < /dev/null &
# open http://192.168.15.86:8080 — paddle detection drawn live by the NPU
```

`infer_npu.py` starts the daemon as a subprocess, waits for it to print `DAEMON_READY`,
keeps the command FIFO open for the whole session, and cleanly sends `"q"` on shutdown.

---

## Step 9 — Benchmark CPU vs NPU (honestly)<a name="step-9"></a>

Measure the *whole* `.detect()` call (preprocess + inference + postprocess) — that's what
the live stream actually pays per frame — for both engines on the same frame.

```bash
# on the board (stop the web server first — it contends for the NPU/FIFOs)
python3 yolo/bench_cpu_vs_npu.py
```

Results on the IQ8-275:

| | Latency | Throughput |
|---|---|---|
| CPU (onnxruntime), full pipeline | 42.0 ms | 23.8 FPS |
| NPU, **raw math only** | 1.7 ms | ~576 FPS |
| NPU, **full pipeline** | **25.3 ms** | **39.5 FPS** |

**The NPU wins end-to-end: 1.7× faster than the CPU**, with the identical detection
(CPU box (224,177)-(383,363) p=0.742; NPU (224,177)-(382,363) p=0.746). But getting there
took one fix — and the detour is the most instructive part of the whole project.

### The trap: feeding the chip in the wrong format

Naively, the NPU pipeline came out at ~57 ms — *slower* than the CPU, despite 84×-faster
math. `yolo/diag_npu_timing.py` breaks down the NPU cycle to find out why. The A16W8 model
runs on **16-bit integer activations**, so the float32 camera frame must be quantized to
`uint16` on the way in and dequantized on the way out. The naive path let the SDK
(`populateInputTensors` / `writeOutputTensors`) do that conversion **element by element on
the CPU** — ~307k elements per frame in, ~10k out — which cost ~13 ms in + ~11 ms out:

| stage (naive `g` path) | time |
|---|---|
| write input `.raw` (float32, 1.17 MB) | ~4 ms |
| **wait** (daemon float→uint16 quantize ~13 ms + execute 1.7 ms + uint16→float dequantize ~11 ms) | ~31 ms |
| read output | ~0.4 ms |

Note it is **not** disk I/O: `/tmp` on this board is a RAM `tmpfs`, so the file write is
nearly free. The cost is the per-element conversion loop inside the SDK.

### The fix: convert in vectorized numpy, hand the chip its native bytes

The conversion math is simple and identical for every element —
`q = round(x/scale − offset)` (clamp to `[0, 65535]`) and back `x = scale·(q + offset)`,
where `scale`/`offset` are the tensor's quantization parameters. So we move it out of the
per-element SDK loop and into **vectorized numpy** (the whole frame at once, sub-millisecond),
and add a `'r'` (raw) command to the daemon that **memcpy's the already-native `uint16` bytes
straight into the tensor buffer** and writes the raw `uint16` output back — no conversion in
the hot loop at all. (`web/infer_npu.py` reads `scale`/`offset` from the daemon's `QUANT`
startup banner so numpy matches the SDK's rounding exactly; the box comes out bit-for-bit the
same.) The cycle now:

| stage (raw `r` path) | time |
|---|---|
| quantize frame in numpy (vectorized) | ~8 ms |
| write native `uint16` bytes | ~3 ms |
| **wait** (daemon memcpy + execute 1.7 ms + write native out + FIFO) | ~9 ms |
| read + dequantize in numpy | ~0.8 ms |

**The lesson:** the NPU's raw math was 84× faster all along; the win only appeared once we
stopped feeding it in the wrong format. An accelerator only pays off if the road to it isn't
the bottleneck — and here the bottleneck was neither the compute nor the disk, but a
format-conversion loop that had no business running per-element on the CPU. Feed the chip its
native format and it wins. (For an even tighter path you'd hand the tensor over **shared
memory** — ion/dmabuf — skipping the file entirely; Qualcomm's AI Hub ships YOLOv8
pre-optimized for this chip and solves that transport out of the box — the natural next step.)

---

## The whole thing in four lessons

1. **Data is the ceiling.** A few hundred similar photos → memorization, not understanding
   (Phase A's IoU 0.56 and its real-world blindness).
2. **Fine-tune, don't build from zero.** YOLOv8n from COCO weights → mAP 0.98 and it works
   in the real room (Phase B).
3. **Quantization has traps.** Plain INT8 crushed the score; A16W8 saved it (Step 7.3).
4. **Measure end-to-end, then fix the road.** The 84×-faster NPU first *looked* slower —
   until measurement traced it to a per-element format conversion, not the chip. Fixing that
   made the NPU win 1.7× end-to-end (Step 9).

Every script referenced here is in this repo with English comments explaining the *why*.
Start at [Step 1](#step-1) and go.
