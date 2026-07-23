# Edge Impulse quickstart — paddle detection on the IQ-8275 EVK

*The shortest path from dataset to live NPU inference. No QAIRT toolchain,
no cross-compilation, no C++ daemon. Expected total time: under two hours.*

---

## What you will have at the end

A native aarch64 binary (`.eim`) running on your IQ-8275 EVK, detecting ping-pong
paddles on the HTP (Hexagon Tensor Processor) NPU at **2 ms per inference**.

---

## Prerequisites

| | |
|---|---|
| Edge Impulse account | Free developer tier at [studio.edgeimpulse.com](https://studio.edgeimpulse.com) |
| Board | IQ-8275 EVK, booted, SSH-accessible |
| Dataset | `pingpong-export/` folder from this repo (contains `info.labels` manifest) |
| `scp` | To copy the `.eim` to the board |

---

## Step 1 — Create the project

1. Log in to [Edge Impulse Studio](https://studio.edgeimpulse.com).
2. Click **Create new project**.
3. Name it (e.g. `pingpong-paddle`). Keep it **Developer / private**.
4. Leave the project type as default — it is set when you add the learning block.

![Edge Impulse project dashboard](img/ei_01_project_dashboard.png)

---

## Step 2 — Upload the dataset

Edge Impulse has a full data collection and labelling workflow built into the Studio.
For this quickstart we use an existing export, but note that for your own project you can
record images, label bounding boxes, and manage your dataset entirely within the platform.

1. Left menu: **Data acquisition** → **Add data** → **Upload data**.
2. Upload mode: **Select a folder** (required — individual-file mode ignores the manifest).
3. Choose folder: `pingpong-export/` from this repository.
4. Category: **Automatically split between training and test**.
5. Click **Upload data**.

When it finishes you should see **559 Training / 128 Testing** with bounding boxes drawn
on the paddle images. The ~316 box-free frames load as negatives — that is correct and
expected.

![Uploading dataset: folder mode with info.labels manifest](img/ei_02_upload_data.png)

> **One label, not two.** All boxes must use the same class name (`paddles`). If you see
> two classes in the Data tab (`paddle` and `paddles`), unify them before training. A
> split class halves the training signal for that object.

---

## Step 3 — Build the impulse

1. Left menu: **Impulse design** → **Create impulse**.
2. **Input block:** Image, **320 x 320**, Resize mode: Squash.
3. **Processing block:** Image (Color depth: RGB).
4. **Learning block:** Object Detection → **YOLOv5** (Foundries.io).
5. Click **Save impulse**.

![Impulse design: Image input + YOLOv5 learning block](img/ei_03_impulse_design.png)

> **Why YOLOv5 and not YOLO-Pro?** YOLO-Pro is marked *Developer Preview* and has a
> pretrained checkpoint only at 640x640. Our impulse uses 320x320, which causes a weight
> loading failure at training time. YOLOv5 has a working 320x320 pretrained checkpoint
> and trains cleanly.

---

## Step 4 — Generate features

Left menu: **Image** → **Generate features** → click **Generate features**.

Wait for the job to complete (~1 to 2 minutes). Verify the summary reads:
**559 training items, 1 class (paddles)**.

![Generate features: 559 items, 1 class](img/ei_04_generate_features.png)

---

## Step 5 — Train

1. Left menu: **YOLOv5** → configure:

   | Setting | Value |
   |---|---|
   | Training processor | GPU |
   | Model size | Small (7.2M params) |
   | Training cycles | 60 |
   | Pretrained weights | True |
   | Validation set size | 20% |
   | Profile int8 model | **enabled** |

2. Click **Save & train**.

Training runs on Edge Impulse's servers (GPU worker). It takes roughly 10 to 15 minutes.

![YOLOv5 training settings: GPU, Small, 60 cycles, Profile int8 enabled](img/ei_05_yolov5_settings.png)

Expected result after training:

| Metric | Expected value | What it means |
|---|---|---|
| mAP@0.5 | ~0.98 | Mean Average Precision at IoU 0.5: how accurately and completely the model detects the paddle. 1.0 is perfect. |
| mAP@0.5:0.95 | ~0.46 | Same metric averaged over stricter box-match thresholds. |

![Training result: mAP 0.980](img/ei_06_training_result.png)

---

## Step 6 — Download the deployment binary

1. Left menu: **Deployment**.
2. Search for **"Qualcomm Dragonwing IQ 8275 EVK (AARCH64 with Qualcomm QNN)"**.
3. Click **Build**.
4. Download the `.eim` file (~20 MB).

The `.eim` is a native aarch64 executable that embeds the model, the Edge Impulse Linux
runtime, and the Qualcomm QNN (Qualcomm Neural Network) TFLite delegate. No separate
runtime installation is required on the board.

---

## Step 7 — Copy to the board and run

```bash
# Copy the binary
scp <your-file>.eim root@<board-ip>:/home/weston/pingpong-demo.eim

# Make executable and run
ssh root@<board-ip> "chmod +x /home/weston/pingpong-demo.eim && /home/weston/pingpong-demo.eim"
```

Replace `<board-ip>` with your board's IP address. Default credentials: `root` / `oelinux123`.

On startup the binary prints a JSON handshake showing it is running on the NPU:

```json
{
  "inferencing_engine": {"engine_type": 4, "properties": ["qnn_delegates"]},
  "model_parameters": {
    "image_input_height": 320, "image_input_width": 320,
    "labels": ["paddles"], "model_type": "object_detection"
  }
}
```

`engine_type: 4` with `qnn_delegates` confirms the model is on the HTP NPU via the QNN
TFLite delegate.

---

## Step 8 — Benchmark

Expected numbers on the IQ-8275 EVK (10-run average):

| Metric | Value |
|---|---|
| Warm-up (JIT compile, first inference) | ~3 ms |
| Steady-state inference | **~2 ms** |
| Equivalent FPS | ~500 fps |

The 3 ms warm-up is a one-time cost on the first call while the QNN delegate compiles the
graph for the HTP. Every subsequent inference is 2 ms.

> **Protocol note.** The `.eim` communicates over a UNIX socket. The correct message key
> for inference is `classify_shm` with features passed via POSIX shared memory. The more
> obvious `classify` key returns an error. See `docs/EDGE_IMPULSE.md` in this repo for
> a full benchmark script.

---

## What is happening under the hood

The `.eim` uses the JIT (Just-In-Time) deployment path: the model is stored as an INT8
TFLite graph and compiled for the HTP NPU on the first inference call. This trades a
small first-call cost (3 ms) for portability — the same binary runs on any QNN-capable
board without recompilation.

The alternative is the AOT (Ahead-Of-Time) path: compile the model offline with the QAIRT
toolchain and ship a chip-specific context binary (`.bin`) that has zero warm-up but only
runs on the exact HTP hardware version it was compiled for. Both paths reach 2 ms
steady-state. The full QAIRT pipeline is documented in [REPRODUCE.md](../docs/REPRODUCE.md).

---

## Next steps

- **Understand everything under the hood:** [BLOG.md](BLOG.md) covers the full story from
  dataset to NPU, including why INT8 alone breaks confidence scores and what the benchmark
  surprise revealed about format conversion.

- **Reproduce the full manual pipeline:** [REPRODUCE.md](../docs/REPRODUCE.md) — every
  command and every script, from dataset to C++ daemon.

- **Platform vs. manual comparison table:** [docs/EDGE_IMPULSE.md](../docs/EDGE_IMPULSE.md).
