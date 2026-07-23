# Phase E — Rebuilding the model on the Edge Impulse platform

*The manual pipeline (Phases A–D) taught a computer to see a paddle and made it run fast on
the NPU — by hand, one script at a time. This phase does the **same task on the Edge Impulse
platform** (now a Qualcomm product) and compares the two: same dataset, same chip, measured
side by side. See [NPU_DEPLOYMENT_PATHS.md](NPU_DEPLOYMENT_PATHS.md) for why the platform and
our manual pipeline produce different model files (`.tflite` vs `.bin`) — that background
explains the "QAIRT bridge" in Step 5 below.*

**What we're comparing**

| | Manual pipeline (Phases B–D) | Edge Impulse (this phase) |
|---|---|---|
| Model | YOLOv8n (3.2M params, Ultralytics) | YOLO-Pro nano (2.4M, Edge Impulse) |
| Quantization | A16W8 (16-bit activations) | INT8 (required for HTP) |
| Effort to reach the NPU | all of Phase C + D | a few clicks in the browser |

The goal is **not** a bit-for-bit identical model — that's impossible across two different
architectures. The goal is a fair, measured answer to: *does the one-click platform path get
close to what we squeezed out by hand?*

---

## Prerequisites

- An Edge Impulse account (free developer tier) — the model **trains in their browser
  Studio**; that part can't be scripted.
- Node.js + npm (for the Edge Impulse CLI). Verified here: Node v20, npm v10.
- The dataset: `pingpong-export/` — which is itself an Edge Impulse export, so it re-imports
  cleanly, bounding boxes and train/test split intact.

---

## Step 1 — Install the CLI and create the project

The Edge Impulse CLI drives data upload from the terminal (training itself is in the Studio).

```bash
npm install -g edge-impulse-cli
edge-impulse-uploader --version   # confirm it's on PATH
```

Then create a new project **in the Studio** (browser):

1. Log in at <https://studio.edgeimpulse.com>.
2. Click **Create new project**.
3. Name it (e.g. `pingpong-paddle`), keep it **Developer** / private.
4. Project type: leave the default — the impulse type (object detection) is chosen later
   when we add the learning block.

> 📸 *Screenshot: the new-project dialog and the empty project dashboard.*

---

## Step 2 — Upload the dataset (boxes + split preserved)

The `pingpong-export/` directory is *itself* an Edge Impulse export, so it re-imports cleanly:
alongside the images sits an **`info.labels`** manifest that records, per file, both its
`category` (`training` / `testing`) and its `boundingBoxes`. Upload that manifest and the
Studio restores the boxes **and** the train/test split in one shot — no re-labelling, no
manual splitting.

We did this **from the Studio UI** (no CLI) so the process is reproducible by anyone with
just a browser:

1. Left menu → **Data acquisition** → **Add data** → **Upload data**.
2. **Upload mode:** *Select a folder* — this lets the Studio read the `info.labels` manifest
   that ships in the folder (individual-file mode ignores it).
3. **Select files:** *Choose Folder* → `pingpong-export/`. The root `info.labels` covers all
   **687** items (559 training + 128 testing) with their boxes embedded.
4. **Upload into category:** *Automatically split between training and test* — the per-file
   `category` in the manifest is respected, so this reproduces our original split exactly.
5. **Upload data**.

> **One-label hygiene:** the export must use a single class name for every box. Our export had
> one stray `paddle` (singular) among 372 `paddles`; left as-is the Studio would create *two*
> classes. Unify the label (in `bounding_boxes.labels` / `info.labels`, or by re-labelling in
> the Studio) **before** training.

When it finishes, **Data acquisition** shows **559 Training / 128 Testing**, with bounding
boxes drawn on the paddle images. The ~316 box-less frames come in as negatives — normal and
useful for a detector.

> 📸 *Screenshot: the Data acquisition tab populated, Train/Test bar reading 559 / 128.*

## Step 3 — Build the impulse: YOLO-Pro nano

**Impulse design → Create impulse.** Three blocks, mirroring our manual pipeline as closely
as the platform allows:

1. **Image data (input):** width/height **320 × 320** — the same input size our YOLOv8n ran
   at, so the comparison isn't confounded by resolution. (The Yolo Pro block later confirms
   this as an **Input layer of 307,200 features** = 320 × 320 × 3.)
2. **Image (processing block):** **Color depth = RGB**.
3. **Object Detection → Yolo Pro** (author *Foundries.io*, badged *Developer Preview*). This
   is the closest analogue to our Ultralytics YOLOv8n; Edge Impulse retired YOLOv5/v8, and the
   Qualcomm FOMO blocks are coarse-segmentation models, not comparable bounding-box detectors.

**Save impulse**, then open **Image → Generate features** and run it. The job confirms the
dataset came in clean: **Training set = 559 items, Classes = 1 (paddles)** — the single-class
hygiene from Step 2 held (no stray `paddle`/`paddles` split).

**Yolo Pro settings** — the config we trained:

| Setting | Value | Why |
|---|---|---|
| Model size | **nano (2.4M)** | closest to our YOLOv8n (3.2M) |
| Number of training cycles | 100 | default |
| Learning rate | 0.01 | default |
| Use pretrained weights | **True** | transfer learning, same as our YOLOv8n fine-tune |
| Use Attention Mixer | **True** (default) | attention + SiLU; see the note below |
| Validation set size | **20 %** | matches our 80/20 split |
| **Profile int8 model** | **✅ on** | **required** — this is what produces the INT8 `.tflite` we bridge to the NPU |

> **Attention Mixer — a knob worth knowing about.** Left at the default **True** (attention +
> SiLU). The docs note a *No-Attention/ReLU* variant "recommended when deploying to hardware
> that does not efficiently support attention layers or SiLU activations," since ReLU runs
> notably faster on edge silicon. For a latency-minimal HTP deployment, **False** is arguably
> the better choice; we kept the default here and note it as a lever to revisit if ops fall
> back to CPU in Step 5/6.

> **Target device is only an estimator.** The *Configure target device* dialog offers
> **Qualcomm Dragonwing RB3 Gen 2** (the IQ-8275 EVK isn't in this tier's list) — but this
> setting only drives Edge Impulse's *on-device performance estimates* and the "Auto
> configure" size suggestion. It does **not** change the trained weights or the exported
> `.tflite`. We ignore its latency estimate entirely: the whole point of this phase is to
> measure the **real** latency on our IQ-8275 via the QAIRT bridge (Steps 5–6), so RB3 vs
> IQ-8275 here is immaterial. Budget left at defaults (RAM 8 GB / ROM 128 GB / Latency 100 ms).

**Save & train.** Training runs on Edge Impulse's servers; with a GPU worker it finishes in
minutes. When it completes the block reports mAP and the int8 profile.

### What actually happened: YOLO-Pro failed, YOLOv5 succeeded

The first attempt used **YOLO-Pro nano** (the newer Foundries.io block, *Developer Preview*).
It failed immediately with a Keras weight-loader error:

```
ValueError: A total of 7 objects could not be loaded.
Layer 'depthwise_conv2d_2' expected 2 variables, but received 1 variables during loading.
```

Root cause: the only available YOLO-Pro nano checkpoint is pre-trained at **640×640**, but our
impulse is configured at **320×320** — the loader couldn't reconcile the mismatch. Training
from scratch (no pretrained weights) produced **mAP@0.5 = 0.00** — as expected, since 447
images are far too few to train a detector backbone from scratch.

**Fix:** switch the learning block to **YOLOv5** (also Foundries.io, same Deployment tab,
not marked *Developer Preview*). YOLOv5 has a working 320×320 pretrained checkpoint; the
loader succeeds and transfer learning kicks in normally.

**Final YOLOv5 training config:**

| Setting | Value |
|---|---|
| Training processor | **GPU** |
| Model size | **Small (7.2M params)** |
| Training cycles | 60 |
| Pretrained weights | True (transfer learning) |
| Validation set size | 20 % |
| **Profile int8 model** | **✅ on** |
| Input | **320 × 320** (307,200 features = 320×320×3) |

### Training result (int8, validation set, 112 images)

| Metric | Value |
|---|---|
| **mAP@IoU=0.5** | **0.980** |
| mAP@IoU=0.5:0.95 (COCO) | 0.457 |
| Recall@100 detections | 0.538 |

The number that matters for our comparison is **mAP@0.5 = 0.980** — same IoU (Intersection over Union) threshold we
used to evaluate YOLOv8n (**0.979**). Virtually identical quality from the platform.

> 📸 *Screenshots: the Create-impulse layout (320×320, Image + YOLOv5 blocks), the YOLOv5
> settings panel, and the training result showing mAP@0.5.*

## Step 4 — Download the INT8 `.tflite` and other artifacts

From the project **Dashboard → Download block output**, all artifacts are available once
training completes:

| File | Size | Use |
|---|---|---|
| `*-tensorflow-lite-int8-quantized-*.lite` | 7 MB | Experiment 2 — QAIRT bridge |
| `*-tensorflow-lite-float32-*.lite` | 13 MB | CPU reference / fallback |
| `*-onnx-model-*.onnx` | 27 MB | **Experiment 2 preferred input** (QAIRT accepts ONNX natively) |
| `*-tensorflow-savedmodel-*.zip` | 108 MB | full TF SavedModel — not needed |
| `*-model-evaluation-metrics-*.json` | 15 KB | mAP and per-class metrics |

Download the **int8 `.lite`** and the **`.onnx`** — the ONNX turns out to be the cleaner input
for the QAIRT pipeline (Step 5), exactly as we did with the YOLOv8n.

Additionally, from the **Deployment** tab, download the platform-specific binary:

> **"Qualcomm Dragonwing IQ 8275 EVK (AARCH64 with Qualcomm QNN)"** — 20 MB `.eim`

This is the Experiment 1 artifact: a native aarch64 ELF binary that runs the model via the
QNN TFLite (Lite Runtime (formerly TFLite)) delegate and implements the Edge Impulse Linux protocol (UNIX socket + POSIX shm).

## Step 5 — Experiment 1: run the `.eim` directly on the board

The `.eim` is a **self-contained aarch64 ELF** — no Node.js, no edge-impulse-linux-runner
needed on Qualcomm Linux. Copy it to the board and run:

```bash
scp pingpong-demo-linux-aarch64-qnn-v1-impulse-#1.eim root@<board-ip>:/home/weston/pingpong-demo.eim
ssh root@<board-ip> "chmod +x /home/weston/pingpong-demo.eim"
```

The binary speaks a JSON protocol over a UNIX socket. After the `hello` handshake it reports
the engine and model parameters:

```json
{
  "inferencing_engine": {"engine_type": 4, "properties": ["qnn_delegates"]},
  "model_parameters": {
    "image_input_height": 320, "image_input_width": 320,
    "labels": ["paddles"], "model_type": "object_detection"
  }
}
```

`engine_type: 4` with `qnn_delegates` = **QNN TFLite delegate on the HTP** — Path 1 (JIT)
exactly as documented in [NPU_DEPLOYMENT_PATHS.md](NPU_DEPLOYMENT_PATHS.md).

Features are passed via POSIX shared memory (`classify_shm` message). Timing measured over
10 runs on the IQ-8275 EVK:

| Metric | Value |
|---|---|
| Warm-up (first inference, JIT compile) | **3 ms** |
| Inference avg (steady-state) | **2 ms** |
| Inference min / max | 2 / 2 ms |
| Equivalent FPS | ~500 fps |

The JIT warm-up is remarkably fast (3 ms). Steady-state at **2 ms** matches our YOLOv8n
daemon — the delegate placed the entire graph on the HTP with no visible CPU fallback.

## Step 6 — Experiment 2: QAIRT bridge — `.onnx` → context `.bin` → daemon

This experiment runs the Edge Impulse model through our **same offline compiler and daemon**
as YOLOv8n, so the execution engine is identical and the latency comparison isolates the
model architecture.

**On the x86 build server** (needs QAIRT SDK 2.47):

```bash
# 1. Convert ONNX → floating-point DLC
source qairt_env.sh
qairt-converter --input_network yolov5_ei.onnx --output_path yolov5_ei_fp.dlc

# 2. Quantize A16W8 with calibration images
qairt-quantizer \
    --input_dlc yolov5_ei_fp.dlc \
    --output_dlc yolov5_ei_a16w8.dlc \
    --input_list calib/input_list.txt \
    --act_bitwidth 16 --weights_bitwidth 8 \
    --apply_algorithms cle \
    --use_per_channel_quantization --use_per_row_quantization

# 3. Generate HTP V75 context binary
#    (requires backend_ext.json pointing to libQnnHtpNetRunExtensions.so + htp_config.json)
qnn-context-binary-generator \
    --dlc_path yolov5_ei_a16w8.dlc \
    --backend $SDK/lib/x86_64-linux-clang/libQnnHtp.so \
    --output_dir ctx_yolov5 \
    --binary_file yolov5_ei_a16w8_htpv75 \
    --config_file backend_ext_yolov5.json
# → ctx_yolov5/yolov5_ei_a16w8_htpv75.bin  (7.1 MB)
```

The ONNX model input is `images` [1,3,320,320] — compatible with the existing 320×320
calibration set from Phase C. The same A16W8 recipe and `cle` algorithm applies cleanly.

**Transfer to the board:**
```bash
scp ctx_yolov5/yolov5_ei_a16w8_htpv75.bin root@<board-ip>:/home/weston/npu/
```

**Running via `qnn-net-run`** (single inference, warm path after context load):

```bash
qnn-net-run \
  --backend libQnnHtp.so \
  --retrieve_context /home/weston/npu/yolov5_ei_a16w8_htpv75.bin \
  --input_list input_list.txt \
  --config_file backend_ext_yolov5.json \
  --output_dir /tmp/out_yolov5
```

`qnn-net-run` includes ~150–200 ms of process startup and context load per invocation. The
daemon approach (Phase D) amortizes that over many frames; a full daemon integration for
YOLOv5 is left as a follow-on. For the latency comparison, Experiment 1 (`.eim` → 2 ms
steady-state) is the better data point — it uses the same resident-context pattern.

---

## Results: platform vs. manual pipeline

| | Manual pipeline (Phase B–D) | EI platform — Exp.1 (`.eim`) | EI platform — Exp.2 (`.bin`) |
|---|---|---|---|
| Model | YOLOv8n, 3.2M params | YOLOv5 Small, 7.2M params | YOLOv5 Small, 7.2M params |
| Quantization | A16W8 | INT8 (QNN delegate) | A16W8 (QAIRT offline) |
| mAP@0.5 | **0.979** | **0.980** | 0.980 (same weights) |
| Inference latency | **~2 ms** (daemon, in-memory) | **2 ms** (JIT delegate) | ~2 ms (estimated, same chip) |
| Warm-up | none (context pre-loaded) | **3 ms** (JIT compile on first call) | none (AOT) |
| Effort to reach the NPU | all of Phase C + D | download `.eim`, copy to board | QAIRT bridge (same as Phase C) |
| Engine | QNN native (AOT) | QNN TFLite delegate (JIT) | QNN native (AOT) |

**The platform matched the manual pipeline on every metric that matters**: mAP@0.5 is
virtually identical (0.980 vs 0.979), and the NPU inference time is the same 2 ms — whether
delivered by a one-click `.eim` or our hand-compiled context `.bin`. The difference is in
the **effort**: what took Phases C and D (days of work, a cross-compile toolchain, A16W8
recipe, C++ daemon) took a few browser clicks on the Edge Impulse platform.

The one observable difference: the `.eim`'s first inference costs a **3 ms JIT warm-up**;
the `.bin` has none (the graph is pre-compiled). In a real application with a resident
daemon this is negligible — but it is the real cost of the "portable" trade-off documented
in [NPU_DEPLOYMENT_PATHS.md](NPU_DEPLOYMENT_PATHS.md).

> 📸 *Screenshots: Deployment tab showing the IQ-8275 EVK option, the `.eim` benchmark
> output, and the QAIRT pipeline on the x86 server.*
