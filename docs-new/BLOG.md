# Teaching a Qualcomm chip to spot a ping-pong paddle

*A practical journey through edge AI — from first dataset photo to live NPU inference,
with two paths: the easy one and the one that teaches you everything.*

---

A tiny glossary for newcomers:

- **Dataset** — example photos plus the answer for each one.
- **Model** — the program that learns patterns from the dataset.
- **Training** — showing examples to the model until it improves.
- **Inference** — using the trained model on a new image.
- **CPU** — the normal general-purpose processor.
- **NPU** — a special processor built to run AI math faster and with less power.
- **Bounding box** — a rectangle drawn around an object in a photo.

---

## What this is about

I wanted a small edge board — a **Qualcomm IQ-8275 EVK**, the kind of chip that goes
inside phones and cameras — to look at a live video feed and draw a box around a
ping-pong paddle in real time.

Simple to describe. Less simple to do. And the *how* you get there turns out to be
more instructive than the destination.

There are two paths to that destination. This story covers both.

---

## Chapter 0 — The fast path: Edge Impulse

*This chapter covers the easiest way to go from "I have images" to "it runs on the NPU."
If you want to understand what's happening under the hood, keep reading past this chapter.
If you just want it to work, this is your chapter.*

**[Edge Impulse](https://edgeimpulse.com)** is a machine-learning platform — now a
Qualcomm product — built precisely for the kind of edge-hardware AI deployment this project
targets. Instead of writing training code, setting up quantization pipelines, and
cross-compiling C++ daemons, you click through a browser UI and download a file that
runs on your board. The headline result of this chapter: **2 ms inference on the NPU**
without touching a command line.

Here's the full sequence:

**1. Create a project and upload the dataset.**
Log in to Edge Impulse Studio, create a project, go to *Data acquisition → Add data →
Upload data*, select the `pingpong-export/` folder. The manifest file inside it
preserves every bounding box and the original train/test split — 559 training images and
128 test images appear fully labelled in seconds.

**2. Build the impulse.**
In *Impulse design*, add an *Image* input (320 × 320, RGB), an *Image* processing block,
and a *FOMO/YOLOv5/YOLOv8* object-detection learning block. A word of experience: the
YOLO-Pro block (Developer Preview) has a checkpoint-size mismatch bug at 320×320 that
silently produces mAP 0.00 — switch to **YOLOv5**, which has a working pretrained
checkpoint at that resolution.

**3. Generate features and train.**
Click *Generate features*, then configure the YOLOv5 block (Small, 60 cycles, GPU,
pretrained weights on) and hit *Save & train*. Training runs on Edge Impulse's servers.
Result: **mAP@0.5 = 0.980** — virtually identical to the hand-built pipeline covered
in the chapters below.

**4. Deploy to the board.**
Go to *Deployment*, find **"Qualcomm Dragonwing IQ 8275 EVK (AARCH64 with Qualcomm QNN)"**,
download the `.eim`. That file is a self-contained native binary — no Node.js, no runtime
to install separately.

**5. Copy and run.**

```bash
scp pingpong-demo.eim root@<board-ip>:/home/weston/
ssh root@<board-ip> "chmod +x /home/weston/pingpong-demo.eim && /home/weston/pingpong-demo.eim"
```

Benchmark on the IQ-8275 EVK (10 runs, `classify_shm` protocol, POSIX shared memory):

| Metric | Value |
|---|---|
| Warm-up (JIT compile, first call) | 3 ms |
| Steady-state inference | **2 ms** |
| Equivalent FPS | ~500 fps |

The `.eim` uses the **QNN TFLite delegate** to place the graph on the HTP (Hexagon Tensor
Processor) NPU automatically. No manual quantization. No context binary compiler. No
cross-compile toolchain. Two milliseconds.

> **The honest note:** all of this worked, including an AI coding agent helping navigate
> the UNIX socket protocol the `.eim` speaks (the correct message key, `classify_shm`, was
> found by reading strings out of the binary). The point isn't that it's trivial — it's
> that the hard parts are handled by the platform. More on AI-assisted development later.

---

## Want to understand what just happened?

The `.eim` binary Edge Impulse gave you contains a quantized neural network model,
a custom runtime, and a QNN delegate — all packaged and compiled for your chip.
In four browser clicks you got the same 2 ms result that the manual pipeline (below)
reaches after days of toolchain setup. That's what the platform buys you.

But *why* does quantization matter? What is a QNN delegate? Why did plain INT8 fail on
a similar model? What is a context binary, and why does the AOT-compiled version have
zero warm-up while the `.eim` needs 3 ms on the first call?

If those questions interest you, the chapters below answer all of them — by showing
exactly what went wrong, why it went wrong, and what the fix revealed.

---

## Part 1 — The dataset: where it all starts

A computer doesn't know what a paddle is. You have to *show* it — hundreds of times —
and mark where the paddle is in each photo. That collection of labelled examples is
the **dataset**, and it is the single most important ingredient. Everything else flows
from its quality.

The dataset for this project: roughly 670 photos of ping-pong paddles in various
settings — on tables, in hands, near and far, in different rooms and lighting conditions.
Each photo has a bounding box hand-drawn around the paddle (or is a "negative" with no
paddle at all, which teaches the model what "nothing" looks like). Photos were labelled
using Edge Impulse's free data collection and labelling tools — one of the few parts of
the platform where even the manual-pipeline path starts.

> **The one thing to remember from Part 1:** more varied data beats a cleverer model
> almost every time. This becomes the clearest lesson of the whole story.

---

## Part 2 — Building a brain from scratch (and why it half-failed)

The first attempt was a custom **CNN (Convolutional Neural Network)** built from zero:
a small network with a single image input, trained only on those ~670 photos.

It worked — on paper. On photos that looked like the training set (person standing back,
good light, full paddle visible) it detected the paddle cleanly. Then a webcam pointed
at a close-up face in a dim room returned nothing. Not a low-confidence detection — nothing.

A diagnostic sequence narrowed it down:

- Made the paddle small, then big → no difference. Not a scale problem.
- Close-up face cropped out → suddenly detected. **It was the face.**
- Dim image brightened → confidence jumped. **Darkness hurt too.**

The from-scratch model hadn't learned "paddle" — it had memorised the photo album.
The album didn't include close-up-face-in-a-dark-room, so neither did the model's world.

Accuracy score: roughly **0.56** (out of 1.0).

> **The one thing to remember from Part 2:** a small model trained on a small,
> similar-looking pile of photos memorizes, it doesn't understand. It looks smart until
> it sees something outside the album.

---

## Part 3 — Fine-tuning a giant's shoulders

The fix: **don't start from zero.** Researchers have trained enormous networks on millions
of everyday photos — every lighting, every distance, every scene. One free family of these
is **YOLO** ("You Only Look Once"), a detector family that spots objects in a single pass
through an image.

The version used here: **YOLOv8n** (nano — the smallest variant, deliberately, because
it has to fit on a tiny chip later). It already knows edges, shapes, faces, and what
"an object held in a hand" looks like. All it needed to learn was the final word: *paddle*.
That teaching — fine-tuning — took the same few hundred photos and a short training run
on a Mac with an MPS (Metal Performance Shaders) GPU backend.

Result: **mAP@0.5 = 0.979** — and it found the paddle in the close-up-face-in-a-dark-room
shot that had completely stumped the from-scratch model.

Same photos. Same computer. Same amount of effort. The only change: starting from a model
that had already seen the world.

> **The one thing to remember from Part 3:** don't build from zero when giants have done
> 99% of the work for free. Fine-tuning a pre-trained model is almost always the right
> starting point.

---

## Part 4 — Watching it live

A number in a spreadsheet is abstract. A live camera feed with a box drawn around the
paddle as it moves — that's proof. A small web server serves an MJPEG stream; each frame
runs through the model on the Mac's CPU, the box is painted, and it appears in a browser
tab. You pick your camera, hit start, wave the paddle, and the box follows it.

The useful design decision here: the server doesn't know which model is running. Whether
it's the from-scratch CNN, the YOLOv8n on CPU, or eventually the NPU on the edge board —
the video layer never changes. Swapping the engine is one line. That decoupling made every
step afterward easier.

---

## Part 5 — Moving to the Qualcomm board

Everything so far ran on a Mac. The real target was the **IQ-8275 EVK** — a credit-card-
sized Qualcomm Linux board running at the edge. The training stays on the big computer
(PyTorch + MPS); the inference moves to the board.

Moving the model means exporting it to **ONNX (Open Neural Network Exchange)** — a
portable format the board can read without the full training stack. The board's ordinary
CPU ran it at roughly 24 fps, smooth enough for live video.

But the board has an NPU sitting unused. That's what it's *for*.

---

## Part 6 — The NPU, the trap, and the lesson that made it worth it

Getting a model onto Qualcomm's **HTP (Hexagon Tensor Processor)** NPU is not a copy-paste
operation. The chip runs faster and uses less power by using *rougher numbers* — rounding
long decimals down to compact integers. This is **quantization**, and it is how edge NPUs
achieve their speed.

### The INT8 trap

First attempt: quantize everything to 8-bit integers (INT8). The model loaded on the NPU.
It detected the paddle. But the confidence score — the number between 0 and 1 that says
"I'm 87% sure this is a paddle" — **collapsed to zero**. The NPU reported detections, but
0% confidence on all of them, which is useless.

The cause is subtle: bounding-box coordinates are large numbers (like 400, 580). Confidence
scores are tiny numbers (0.87). Forcing both through the same 8-bit scale — calibrated for
the large coordinates — crushes the delicate confidence values into nothing. Same number
scale, very different magnitudes.

The fix: **A16W8** — keep weights at 8-bit (compact) but allow activations (the
in-flight numbers) to use 16-bit precision. That extra range protects the confidence score.
The NPU reported a healthy 87% detection immediately.

This is the **QAIRT (Qualcomm AI Runtime SDK)** pipeline:
1. `qairt-converter`: ONNX → floating-point DLC (Deep Learning Container)
2. `qairt-quantizer`: DLC → A16W8 DLC (with calibration images)
3. `qnn-context-binary-generator`: DLC → chip-specific context `.bin`

The context `.bin` is compiled ahead-of-time for the exact HTP version on this board. No
JIT warm-up, no fallback, no portability — but maximum speed and minimum startup cost.

### The benchmark surprise

Raw NPU math: **1.74 ms per inference** vs **145 ms on CPU** — 84× faster. The live
stream should fly.

It didn't — at first. The end-to-end measurement showed the NPU version at ~57 ms per
frame versus ~42 ms on CPU. The 84×-faster engine was losing the race.

The investigation found the cause: **format conversion**. The NPU wants its input as
compact 16-bit integers in a specific layout. The camera delivers float32 RGB. Something
has to translate every frame — 307,200 numbers per frame — and the default toolkit path
was doing it one number at a time on the CPU. *That* was the missing 30 ms.

Fix: convert the entire frame at once with a vectorized math call (a fraction of a
millisecond), then hand the chip exactly the bytes it expects.

After that fix: **NPU: 25 ms/frame vs CPU: 42 ms/frame — a genuine 1.7× end-to-end win**.

> **The one thing to remember from Part 6:** a faster engine only wins if the road to it
> isn't the bottleneck. The NPU's math was 84× faster from the start; the win only
> appeared after fixing the format conversion that was feeding it one number at a time.

---

## Part 7 — The daemon: a resident NPU process

The way the benchmarks are measured above — loading the context binary fresh per call —
includes process-startup overhead (~150 ms). For a live video application you want the
model *resident*: loaded once, accepting frames forever.

The solution: a **C++ daemon** that holds the QNN context in memory, reads raw frames from
a named FIFO pipe, runs inference, and pushes results back — while a second thread encodes
and streams MJPEG over HTTP. The browser tab stays live; the NPU never sleeps between
frames.

This is the architecture the 25 ms/frame number above comes from. Frame arrives over FIFO,
format-converts in ~0.2 ms, NPU runs in ~1.74 ms, box coordinates decode and the HTTP
encoder sees them. End to end: ~25 ms, ~40 fps.

---

## Part 8 — The comparison

| | Manual pipeline (Parts 2–7) | Edge Impulse (Chapter 0) |
|---|---|---|
| Model | YOLOv8n, 3.2M params | YOLOv5 Small, 7.2M params |
| Quantization | A16W8 (manual recipe) | INT8 via QNN TFLite delegate |
| mAP@0.5 | **0.979** | **0.980** |
| NPU inference | **~2 ms** (daemon, resident) | **2 ms** (JIT delegate) |
| Warm-up cost | None (AOT context pre-loaded) | 3 ms (JIT on first call) |
| Path to get here | Parts 2–7 of this story | Chapter 0 |
| Reproducible with | [REPRODUCE.md](../docs/REPRODUCE.md) | [QUICKSTART_EI.md](QUICKSTART_EI.md) |

The platform matched every metric that matters. **2 ms either way, mAP within 0.001.**

The gap is in the journey: the manual pipeline required a cross-compile toolchain, a
quantization recipe, a custom C++ daemon, and a week of debugging. Edge Impulse required
a browser and an afternoon.

Why does the manual path exist in this story? Because taking it taught things the
platform hides:

- Why INT8 silently breaks a detector's confidence score and how A16W8 fixes it.
- Why a chip that's 84× faster can lose an end-to-end race, and what actually wins it.
- That JIT warm-up (3 ms) and AOT startup (0 ms) are different deployment trade-offs,
  not just different tools.
- That a USB camera sometimes needs a full power-cycle, not a replug, to come back — and
  other unglamorous hardware realities that no platform abstracts away.

---

## A note on how this was built

None of this is the work of a lone wizard. Throughout the project, an **AI coding agent
(Claude Code)** was the co-pilot: writing scripts, explaining toolchain errors, running
benchmarks on the board over SSH, interpreting quantization failures, and suggesting the
correct UNIX socket message key for the `.eim` protocol after reading strings out of
the binary.

That last point is worth dwelling on: when the `.eim`'s `classify` message returned
"Failed to handle message," the agent found `classify_shm` by searching the binary itself —
not from documentation that didn't exist yet.

If you reproduce this project, use an AI agent. The QAIRT toolchain has sharp edges,
error messages that require domain context to decode, and a learning curve that AI
assistance cuts significantly. It doesn't make hard things trivial — but it makes them
tractable in an afternoon rather than a week.

---

## The lessons, short version

1. **Data is your ceiling.** A few hundred similar photos produce a memorizer, not a
   generalizer. More varied examples beat a cleverer model.

2. **Start from a pre-trained model.** Fine-tuning YOLOv8n went from 0.56 to 0.979 mAP
   with the same dataset, same training time, same effort.

3. **Measure the whole pipeline honestly.** The NPU was 84× faster at math but lost the
   end-to-end race until the format-conversion bottleneck was fixed.

4. **The bottleneck is rarely where you expect.** Not the chip. Not the disk. One-number-
   at-a-time format conversion on the CPU.

5. **The platform and the manual path give the same answer.** Edge Impulse reaches 2 ms on
   the NPU in an afternoon. The manual path reaches 2 ms in a week and teaches you why.

---

## Go further

- **The easy path — copy-paste instructions:**
  [QUICKSTART_EI.md](QUICKSTART_EI.md)

- **The deep-dive — every command, every script:**
  [REPRODUCE.md](../docs/REPRODUCE.md)
