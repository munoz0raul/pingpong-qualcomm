# Two ways to put a model on an NPU: offline compile vs runtime delegate

*Background note for [Phase E](REPRODUCE.md), where we rebuild this project's model on the
Edge Impulse platform and compare it to the one we built by hand. Before comparing the two
paths, it's worth understanding **why they even produce different files** — a `.bin` in our
manual pipeline, a `.tflite` on the platform. That difference isn't cosmetic; it's two
philosophies of how a model reaches the silicon.*

---

## The two files, and what they actually are

When people say "the model," they can mean two very different things depending on where you
are in the chain. An analogy with ordinary code makes it concrete:

- **A `.tflite` file is like portable source code** (think a `.c` file). It's the model's
  graph plus its weights, in a neutral format. On its own it does **not** know how to run on
  a Hexagon NPU — it needs something at runtime to translate it for that specific chip.
- **A context `.bin` is like a compiled binary for one exact chip** (think an executable
  built for one CPU). The graph has **already been converted** into commands for the
  Hexagon V75 specifically, ready to load and fire.

So the real question is: **who bridges the gap from portable graph to chip-specific
commands, and *when*?** There are two answers, and they are the two philosophies.

---

## Path 1 — Runtime delegate (JIT): what Edge Impulse does

Edge Impulse ships a `.tflite` (INT8-quantized) inside its `.eim` package. When that package
starts **on the board**, it hands the graph to a runtime library
(`libQnnTFLiteDelegate.so`), which compiles the operations for the Hexagon NPU **right
there, at load time**. It's the same idea as a Just-In-Time compiler: the translation
happens the moment you run it, on the target.

- **Upside:** one `.tflite` runs on *any* target — an x86 laptop, an Android phone, the
  IQ-8275 — with no offline compile step. This is what lets the platform be "one click."
- **Cost:** the delegate decides **at runtime** which operations run on the NPU and which
  fall back to the CPU. If an op isn't supported, it silently reverts to CPU — and you only
  find out by measuring. There's also a warm-up cost on the first inference while it
  compiles.

## Path 2 — Offline compile (AOT): what we did by hand

In our manual pipeline (Phase C), `qnn-context-binary-generator` took the ONNX model,
converted it to a DLC, quantized it (A16W8), and **serialized the graph already compiled for
the V75** into a `.bin`. Every "does this run on the NPU?" decision was made **ahead of
time**, on the x86 box — Ahead-Of-Time compilation.

- **Upside:** deterministic and auditable. We *watched* `QnnGraph_execute ... result 0` and
  know the entire graph is on the NPU, with no hidden fallback. It loads and runs directly,
  with no compile-time warm-up. Minimal latency (1.7 ms of raw math).
- **Cost:** the `.bin` is **married to the V75** — it won't run on any other NPU. And the
  process is manual: it was, quite literally, the work of Phases C and D.

---

## Side by side

| | `.tflite` + delegate (Edge Impulse) | context `.bin` (our manual pipeline) |
|---|---|---|
| When it compiles for the NPU | **at runtime**, on the board (JIT) | **offline**, on the x86 box (AOT) |
| Portability | runs on any target | married to the Hexagon V75 |
| CPU fallback | yes — automatic and silent | no — the whole graph is on the NPU, or it fails |
| Effort to produce | one click | all of Phase C + D |
| Latency profile | + warm-up, + risk of silent fallback | minimal and deterministic |

**Neither is "more correct."** It's the classic trade-off: **convenience and portability
(JIT) vs control and latency (AOT)**. A runtime delegate meets you where you are; an offline
compile wrings out every millisecond but ties you to one chip and a lot of manual work.

---

## Why this matters for the comparison

This is exactly the difference worth measuring: **does the platform's one-click path get
close to the latency we squeezed out by hand?**

To answer it *fairly*, in Phase E we take the `.tflite` Edge Impulse produces and run it
**through the same offline compiler** (QAIRT → `.bin`) that we used for our own model. That
way the comparison isolates the one variable that actually matters — **the model
architecture** (Edge Impulse's YOLO-Pro vs our YOLOv8n) — because the execution engine
becomes identical on both sides.

Without that bridge, we'd be comparing the model *and* the runtime at the same time, and
couldn't tell which one to credit for any difference we saw. With it, we run both models on
the same daemon, on the same chip, and let the numbers speak.
