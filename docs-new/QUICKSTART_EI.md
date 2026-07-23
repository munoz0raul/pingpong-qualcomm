# Edge Impulse quickstart — paddle detection on the IQ-8275 EVK

*The shortest path from dataset to live NPU inference. No QAIRT toolchain,
no cross-compilation, no C++ daemon. Expected total time: under two hours.*

---

## What you'll have at the end

A native aarch64 binary (`.eim`) running on your IQ-8275 EVK, detecting ping-pong
paddles on the HTP NPU at **2 ms per inference**.

---

## Prerequisites

| | |
|---|---|
| Edge Impulse account | Free developer tier — [studio.edgeimpulse.com](https://studio.edgeimpulse.com) |
| Board | IQ-8275 EVK, booted, SSH-accessible |
| Dataset | `pingpong-export/` folder from this repo (contains `info.labels` manifest) |
| `scp` | To copy the `.eim` to the board |

---

## Step 1 — Create the project

1. Log in to [Edge Impulse Studio](https://studio.edgeimpulse.com).
2. Click **Create new project**.
3. Name it anything (e.g. `pingpong-paddle`). Keep it **Developer / private**.
4. Leave the project type as default — it will be set when you add the learning block.

---

## Step 2 — Upload the dataset

1. Left menu → **Data acquisition** → **Add data** → **Upload data**.
2. Upload mode: **Select a folder** (important — individual-file mode ignores the manifest).
3. Choose folder: **`pingpong-export/`** from this repository.
4. Category: **Automatically split between training and test**.
5. Click **Upload data**.

When it finishes you should see **559 Training / 128 Testing** with bounding boxes drawn
on the paddle images. The ~316 box-free frames load as negatives — that's correct.

> **One label, not two.** The manifest must have a single consistent class name (`paddles`).
> If you see two classes (`paddle` and `paddles`) in the Data tab, merge the stray entries
> before training — a mislabelled class halves your training signal for that class.

---

## Step 3 — Build the impulse

1. Left menu → **Impulse design** → **Create impulse**.
2. **Input block:** Image, **320 × 320**, Resize mode: Squash.
3. **Processing block:** Image (Color depth: RGB).
4. **Learning block:** Object Detection → **YOLO v5** (Foundries.io).
   - Do **not** choose YOLO-Pro — it has a checkpoint-size bug at 320×320 that
     produces mAP 0.00.
5. Click **Save impulse**.

---

## Step 4 — Generate features

Left menu → **Image** → **Generate features** → **Generate features**.

Wait for the job to complete (~1–2 min). Verify the banner reads **559 training items,
1 class (paddles)**.

---

## Step 5 — Train

1. Left menu → **YOLO v5** → configure:

   | Setting | Value |
   |---|---|
   | Training processor | GPU |
   | Model size | Small (7.2M params) |
   | Training cycles | 60 |
   | Pretrained weights | **True** |
   | Validation set size | 20 % |
   | Profile int8 model | **✅ enabled** |

2. Click **Save & train**.

Training runs on Edge Impulse's servers (GPU). It takes roughly 10–15 minutes.

Expected result after training:

| Metric | Expected value |
|---|---|
| mAP@0.5 | ~0.98 |
| mAP@0.5:0.95 | ~0.46 |

---

## Step 6 — Download the deployment binary

1. Left menu → **Deployment**.
2. Search or scroll to **"Qualcomm Dragonwing IQ 8275 EVK (AARCH64 with Qualcomm QNN)"**.
3. Click **Build**.
4. Download the `.eim` file (≈20 MB).

---

## Step 7 — Copy to the board and run

```bash
# Copy to the board
scp <your-file>.eim root@<board-ip>:/home/weston/pingpong-demo.eim

# Make it executable
ssh root@<board-ip> "chmod +x /home/weston/pingpong-demo.eim"

# Run it
ssh root@<board-ip> "/home/weston/pingpong-demo.eim"
```

Replace `<board-ip>` with your board's IP (default: `192.168.15.86` if using the same
network as this project). Default credentials: `root` / `oelinux123`.

---

## Step 8 — Run a benchmark

The `.eim` speaks a JSON protocol over a UNIX socket. The quickest check is the hello
handshake it prints on startup:

```json
{
  "inferencing_engine": {"engine_type": 4, "properties": ["qnn_delegates"]},
  "model_parameters": {
    "image_input_height": 320, "image_input_width": 320,
    "labels": ["paddles"], "model_type": "object_detection"
  }
}
```

`engine_type: 4` with `qnn_delegates` confirms the model is running on the HTP NPU
via the QNN TFLite delegate.

Expected benchmark numbers (10-run average on IQ-8275 EVK):

| Metric | Value |
|---|---|
| Warm-up (first inference) | ~3 ms |
| Steady-state inference | **~2 ms** |

> **Tip:** features are passed via POSIX shared memory, not inline in the JSON message.
> Use message key `classify_shm` with `{"elements": N}` in the socket protocol.
> See `docs/EDGE_IMPULSE.md` for the full benchmark script.

---

## What just happened under the hood

The `.eim` contains the model quantized to INT8, packaged with the Edge Impulse Linux
runtime and the Qualcomm QNN TFLite delegate. On first inference the delegate JIT-compiles
the graph for the HTP NPU (3 ms). After that, every call is a resident NPU execution
(2 ms).

This is the JIT (Just-In-Time) deployment path, trading a small first-call cost for
portability — the same binary would run on any QNN-capable board without recompilation.
The alternative AOT (Ahead-Of-Time) path using the QAIRT toolchain is documented in
[REPRODUCE.md](../docs/REPRODUCE.md) — same 2 ms result, more effort, zero warm-up.

---

## Next steps

- **Understand what the platform is doing:** read [BLOG.md](BLOG.md) — it covers every
  layer of this pipeline, including why INT8 alone breaks detector confidence scores and
  what the NPU bottleneck actually was.

- **Reproduce the full manual pipeline:** [REPRODUCE.md](../docs/REPRODUCE.md) — every
  command, every script, from dataset to C++ daemon.

- **Comparison table (platform vs. manual):** see the results section in
  [docs/EDGE_IMPULSE.md](../docs/EDGE_IMPULSE.md).
