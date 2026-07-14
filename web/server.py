#!/usr/bin/env python3
"""
Live detection webserver (Phase A6) — capture webcam -> infer -> draw -> stream to browser.

Runs on the Mac (quick validation) and on the board (same code, only the camera and provider change).
No extra dependencies: stdlib (http.server) + opencv (capture/drawing) + infer.py.

** Architecture: state machine controlled by the browser **
Instead of opening the camera at startup and streaming forever, the server exposes a
small API and only captures while it is "on":

    GET  /                      -> page with dropdowns + start/stop
    GET  /api/cameras           -> [{index, name}]     detected cameras
    GET  /api/modes?camera=N    -> {resolutions, fps}  modes the camera accepts
    POST /api/start  {camera,width,height,fps}         opens the camera and starts
    POST /api/stop                                     releases the camera and stops
    GET  /stream                -> MJPEG multipart (only delivers frames if on)

** How we discover resolutions/fps (the trick) **
OpenCV does NOT list a webcam's modes. So we PROBE: for each candidate resolution/fps,
we ask the camera (cap.set) and read back (cap.get) what it actually
accepted — the camera "snaps" to the nearest supported mode. Deduplicating the
returned values gives us the real list of modes. It's the standard pragmatic method.

** MJPEG stream (multipart/x-mixed-replace) **
The server keeps the HTTP connection open and pushes one JPEG after another; the browser
swaps the displayed image on each chunk -> looks like video. The per-frame loop is:
    capture -> detector.detect() -> detector.draw() -> encode JPEG -> yield

Usage:
    training/.venv/bin/python web/server.py
    # open in the browser: http://localhost:8080
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# PaddleDetector is imported in main() according to --model (cnn -> infer.py, yolo -> infer_yolo.py).
# Both expose the SAME interface (.detect()/.draw()), so the MJPEG loop doesn't change.

# Candidate resolutions/fps we will PROBE (the camera snaps to the nearest).
CANDIDATE_RES = [
    (320, 240), (640, 480), (800, 600), (1024, 768),
    (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160),
]
CANDIDATE_FPS = [15, 24, 30, 60]
MAX_CAMERA_INDEX = 5      # we probe indices 0..5

# OpenCV capture backend. On the Mac the default (CAP_ANY -> AVFoundation) works.
# On the Linux board the default falls into GStreamer, which BREAKS when changing resolution
# after opening; forcing V4L2 (--backend v4l2) resolves it. Filled in main().
CAP_BACKEND = cv2.CAP_ANY


def _open(index):
    """Open the camera with the chosen backend (CAP_ANY on the Mac, V4L2 on the board)."""
    return cv2.VideoCapture(index, CAP_BACKEND)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>pingpong — paddle detection</title>
<style>
  body { background:#111; color:#eee; font-family:sans-serif; text-align:center; margin:0; padding:20px; }
  h1 { font-weight:400; font-size:20px; }
  .controls { display:flex; gap:10px; justify-content:center; align-items:center;
              flex-wrap:wrap; margin:16px 0; }
  select, button { font-size:14px; padding:6px 10px; border-radius:6px;
                   border:1px solid #555; background:#222; color:#eee; }
  button { cursor:pointer; }
  button:disabled { opacity:0.4; cursor:not-allowed; }
  #start { background:#1b5e20; border-color:#2e7d32; }
  #stop  { background:#5e1b1b; border-color:#7d2e2e; }
  #video { width:480px; max-width:95vw; border:2px solid #444; border-radius:8px;
           background:#000; aspect-ratio:16/9; }
  .meta { color:#888; font-size:13px; margin-top:8px; }
  label { color:#aaa; font-size:13px; }
</style></head>
<body>
  <h1>paddle detection &mdash; live</h1>
  <div class="controls">
    <label>camera <select id="camera"></select></label>
    <label>resolution <select id="res" disabled></select></label>
    <label>fps <select id="fps" disabled></select></label>
    <button id="start" disabled>start</button>
    <button id="stop" disabled>stop</button>
  </div>
  <img id="video" src="">
  <div class="meta" id="status">loading cameras&hellip;</div>

<script>
const $ = id => document.getElementById(id);
const statusEl = $("status");

async function loadCameras() {
  const cams = await (await fetch("/api/cameras")).json();
  const sel = $("camera");
  sel.innerHTML = "";
  if (!cams.length) { statusEl.textContent = "no camera found"; return; }
  for (const c of cams) {
    const o = document.createElement("option");
    o.value = c.index; o.textContent = c.name;
    sel.appendChild(o);
  }
  await loadModes();
}

async function loadModes() {
  const cam = $("camera").value;
  statusEl.textContent = "probing modes for camera " + cam + "…";
  $("res").disabled = $("fps").disabled = $("start").disabled = true;
  const m = await (await fetch("/api/modes?camera=" + cam)).json();
  const res = $("res"); res.innerHTML = "";
  for (const [w, h] of m.resolutions) {
    const o = document.createElement("option");
    o.value = w + "x" + h; o.textContent = w + " x " + h;
    res.appendChild(o);
  }
  // default: the largest width <= 1280 (light for CPU), otherwise the first
  const pref = m.resolutions.findIndex(r => r[0] === 1280);
  res.selectedIndex = pref >= 0 ? pref : 0;
  const fps = $("fps"); fps.innerHTML = "";
  for (const f of m.fps) {
    const o = document.createElement("option");
    o.value = f; o.textContent = f + " fps";
    fps.appendChild(o);
  }
  const p30 = m.fps.indexOf(30);
  fps.selectedIndex = p30 >= 0 ? p30 : 0;
  res.disabled = fps.disabled = $("start").disabled = false;
  statusEl.textContent = "ready to start";
}

async function start() {
  const [w, h] = $("res").value.split("x").map(Number);
  const body = { camera: Number($("camera").value), width: w, height: h, fps: Number($("fps").value) };
  statusEl.textContent = "starting…";
  const r = await (await fetch("/api/start", {method:"POST", body: JSON.stringify(body)})).json();
  if (!r.ok) { statusEl.textContent = "error: " + r.error; return; }
  $("video").src = "/stream?t=" + Date.now();   // cache-buster forces reconnect
  $("start").disabled = $("camera").disabled = $("res").disabled = $("fps").disabled = true;
  $("stop").disabled = false;
  statusEl.textContent = `live: ${r.actual.width}x${r.actual.height} @ ${r.actual.fps} fps`;
}

async function stop() {
  await fetch("/api/stop", {method:"POST"});
  $("video").src = "";
  $("stop").disabled = true;
  $("start").disabled = $("camera").disabled = $("res").disabled = $("fps").disabled = false;
  statusEl.textContent = "stopped";
}

$("camera").addEventListener("change", loadModes);
$("start").addEventListener("click", start);
$("stop").addEventListener("click", stop);
loadCameras();
</script>
</body></html>""".encode("utf-8")


def list_cameras(indices=None):
    """Probe a list of indices and return those that open and deliver at least 1 frame.

    On the Mac the webcam is index 0..5. On the board (Linux/V4L2) index N maps to
    /dev/videoN — the Elgato is /dev/video26, so we pass --cameras 26 and avoid
    probing the dozens of ISP nodes that aren't capture cameras.
    """
    if indices is None:
        indices = list(range(MAX_CAMERA_INDEX + 1))
    found = []
    for i in indices:
        cap = _open(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append({"index": i, "name": f"Camera {i}"})
        cap.release()
    return found


def probe_modes(index):
    """Open the camera once and probe resolutions + fps it accepts (set/get)."""
    cap = _open(index)
    if not cap.isOpened():
        return {"resolutions": [], "fps": []}

    resolutions, seen = [], set()
    for (w, h) in CANDIDATE_RES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if aw > 0 and ah > 0 and (aw, ah) not in seen:
            seen.add((aw, ah))
            resolutions.append([aw, ah])
    resolutions.sort()

    fps, seen_f = [], set()
    for f in CANDIDATE_FPS:
        cap.set(cv2.CAP_PROP_FPS, f)
        af = int(round(cap.get(cv2.CAP_PROP_FPS)))
        if af > 0 and af not in seen_f:
            seen_f.add(af)
            fps.append(af)
    cap.release()
    # if the resolution probe returned nothing (backend doesn't respond to set/get),
    # we offer the candidates — start() applies and reads the real value anyway.
    if not resolutions:
        resolutions = [list(r) for r in CANDIDATE_RES]
    # fps via OpenCV is unreliable on some webcams; if the probe didn't yield
    # useful options, we offer the candidate list so the user can choose.
    if len(fps) < 2:
        fps = CANDIDATE_FPS[:]
    fps.sort()
    return {"resolutions": resolutions, "fps": fps}


class StreamState:
    """Owner of the camera: opens/closes on demand, reads serialized by a lock."""

    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        self.index = None
        # bumped on every start(); a stream loop remembers the generation it was
        # serving and only releases the camera if it's still the current one — so a
        # late-dying old connection can't tear down a camera a newer start() just opened.
        self.generation = 0

    def start(self, index, width, height, fps):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            cap = _open(index)
            if not cap.isOpened():
                self.cap = None
                self.running = False
                raise RuntimeError(f"couldn't open camera {index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap = cap
            self.running = True
            self.index = index
            self.generation += 1
            return {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": int(round(cap.get(cv2.CAP_PROP_FPS))) or fps,
                "generation": self.generation,
            }

    def stop(self, generation=None):
        """Release the camera. If `generation` is given, only stop when it's still the
        current session — lets a stream's cleanup fire without clobbering a newer start()."""
        with self.lock:
            if generation is not None and generation != self.generation:
                return
            self.running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def read(self):
        with self.lock:
            if not self.running or self.cap is None:
                return None
            ok, frame = self.cap.read()
            return frame if ok else None

    def active_index(self):
        """Index of the camera in use right now (or None). Used to avoid re-probing a busy device."""
        with self.lock:
            return self.index if (self.running and self.cap is not None) else None


class Handler(BaseHTTPRequestHandler):
    state = None       # injected in main()
    detector = None
    camera_indices = None   # list of indices to probe (None = 0..MAX)

    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path)
        if route.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif route.path == "/api/cameras":
            # if a camera is already in use, report it without re-probing (the device is
            # busy and reopening would fail — this was the cause of "camera not found").
            active = self.state.active_index() if self.state else None
            if active is not None:
                self._json([{"index": active, "name": f"Camera {active} (in use)"}])
            else:
                self._json(list_cameras(self.camera_indices))
        elif route.path == "/api/modes":
            qs = parse_qs(route.query)
            cam = int(qs.get("camera", ["0"])[0])
            self._json(probe_modes(cam))
        elif route.path == "/stream":
            self._stream()
        else:
            self.send_error(404)

    def do_POST(self):
        route = urlparse(self.path)
        if route.path == "/api/start":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length)) if length else {}
            try:
                actual = self.state.start(
                    int(data["camera"]), int(data["width"]),
                    int(data["height"]), int(data["fps"]),
                )
                self._json({"ok": True, "actual": actual})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, code=500)
        elif route.path == "/api/stop":
            self.state.stop()
            self._json({"ok": True})
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.end_headers()
        my_gen = self.state.generation      # the session this stream is serving
        try:
            while self.state.running:
                frame = self.state.read()
                if frame is None:
                    time.sleep(0.02)      # camera stopping/hiccup: don't spin idle
                    continue
                det = self.detector.detect(frame)     # CPU inference on this frame
                self.detector.draw(frame, det)        # green box if there's a paddle
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg.tobytes())
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass       # browser closed the tab / hit stop — normal end
        finally:
            # ALWAYS release the camera when the stream ends — whether the user hit stop,
            # closed the tab, or the connection just dropped. Without this, a closed tab
            # leaves state.running=True and /dev/videoN held open forever (the "camera
            # stuck / not found" bug): the next /api/cameras sees a busy device it can't
            # reopen. Releasing here makes the explicit Stop button optional, not required.
            # Pass my_gen so a late-dying old stream can't close a camera a newer start() opened.
            self.state.stop(generation=my_gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--onnx", default=None, help="path to the .onnx (default depends on --model)")
    ap.add_argument("--model", default="cnn", choices=["cnn", "yolo", "npu"],
                    help="cnn: Phase A CNN (infer.py) | yolo: YOLOv8 CPU (infer_yolo.py) | "
                         "npu: YOLOv8 on the NPU via C++ daemon (infer_npu.py, board only)")
    ap.add_argument("--cameras", default=None,
                    help="camera indices to probe, comma-separated "
                         "(Mac: omit=0..5; board: '26' for /dev/video26)")
    ap.add_argument("--backend", default="auto", choices=["auto", "v4l2"],
                    help="capture backend: 'auto' on the Mac; 'v4l2' on the Linux board "
                         "(the default falls into GStreamer, which breaks when changing resolution)")
    args = ap.parse_args()

    global CAP_BACKEND
    CAP_BACKEND = cv2.CAP_V4L2 if args.backend == "v4l2" else cv2.CAP_ANY

    # choose the engine according to --model; all expose PaddleDetector(.detect()/.draw()).
    if args.model == "yolo":
        from infer_yolo import PaddleDetector
    elif args.model == "npu":
        from infer_npu import PaddleDetector
    else:
        from infer import PaddleDetector

    detector = PaddleDetector(onnx_path=args.onnx)
    print(f"model ({args.model}): {detector.onnx_path}")
    print(f"providers: {detector.sess.get_providers()}")

    Handler.state = StreamState()
    Handler.detector = detector
    if args.cameras:
        Handler.camera_indices = [int(x) for x in args.cameras.split(",")]
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"serving at {url}   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        Handler.state.stop()
        # NPU: shut down the C++ daemon (send 'q' + wait for the process to exit)
        if hasattr(detector, "close"):
            detector.close()


if __name__ == "__main__":
    main()
