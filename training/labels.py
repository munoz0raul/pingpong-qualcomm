#!/usr/bin/env python3
"""
Reading annotations from the Edge Impulse export (Phase A1).

The Edge Impulse export stores the boxes in a `bounding_boxes.labels` file
(a JSON) inside each split (training/ and testing/). The format is:

    {
      "version": 1,
      "type": "bounding-box-labels",
      "boundingBoxes": {
        "file1.jpg": [{"label": "paddles", "x": 188, "y": 9,
                          "width": 79, "height": 138}],
        "file2.jpg": [],            # empty list = image WITHOUT paddle (background)
        ...
      }
    }

Important points about this dataset (see the pingpong-dataset memory):
  - x, y are the TOP-LEFT corner of the box, in ABSOLUTE PIXELS.
  - An image may have 0, 1 or (rarely) 2 boxes. In Phase A the model is a
    "1 box" model, so we pick the LARGEST box when there is more than one.
  - The label comes as "paddles" (most cases) or "paddle" (1 case). We treat
    everything as a single class — what matters in this phase is WHERE the
    paddle is, not WHICH paddle it is.

This module is the single entry point for the annotations. Both the Phase A
preprocess and the Phase B YOLO preparation read the data through here — so the
logic for "how to interpret the export" lives in one place only.
"""

import json
import os

# Export root folder, relative to the project root (training/ is a sibling of pingpong-export/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(_THIS_DIR, "..", "pingpong-export")


class Box:
    """A box in absolute pixels. x,y = top-left corner."""

    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x, y, w, h):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)

    @property
    def area(self):
        return self.w * self.h

    def __repr__(self):
        return f"Box(x={self.x:.0f}, y={self.y:.0f}, w={self.w:.0f}, h={self.h:.0f})"


class Sample:
    """An image from the dataset: path on disk + its boxes (0, 1 or more)."""

    __slots__ = ("path", "boxes")

    def __init__(self, path, boxes):
        self.path = path        # absolute path to the .jpg
        self.boxes = boxes      # list[Box] (empty = background image)

    @property
    def has_paddle(self):
        return len(self.boxes) > 0

    @property
    def main_box(self):
        """The LARGEST box (Phase A 1-box model), or None if it is background."""
        if not self.boxes:
            return None
        return max(self.boxes, key=lambda b: b.area)


def load_split(split):
    """
    Read a split ("training" or "testing") and return list[Sample].

    Only includes images that physically exist on disk AND appear in
    bounding_boxes.labels (the labels sometimes lists a few files more/fewer
    than the folder — the intersection is the real dataset).
    """
    split_dir = os.path.join(EXPORT_DIR, split)
    labels_path = os.path.join(split_dir, "bounding_boxes.labels")
    with open(labels_path) as f:
        data = json.load(f)

    bboxes = data["boundingBoxes"]
    samples = []
    for filename, raw_boxes in bboxes.items():
        img_path = os.path.join(split_dir, filename)
        if not os.path.exists(img_path):
            continue  # file listed but missing from the folder — ignore
        boxes = [Box(b["x"], b["y"], b["width"], b["height"]) for b in raw_boxes]
        samples.append(Sample(img_path, boxes))
    return samples


def _report(split, samples):
    n = len(samples)
    com = sum(1 for s in samples if s.has_paddle)
    print(f"[{split}] {n} images  |  with paddle: {com}  |  background: {n - com}")


if __name__ == "__main__":
    # Sanity check: run `python labels.py` and compare the counts vs. expected
    # (training ~539: 295 with / 244 background ; testing ~124: 76 with / 48 background).
    for split in ("training", "testing"):
        samples = load_split(split)
        _report(split, samples)
        # show 1 example with a box
        for s in samples:
            if s.has_paddle:
                print(f"        example: {os.path.basename(s.path)}  ->  {s.main_box}")
                break
