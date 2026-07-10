#!/usr/bin/env python3
"""
PaddleDetNet — the 1-box detection network (Phase A3).

The idea in one sentence: the 320x320 image comes in, passes through a stack of
convolutions that progressively "see" larger shapes, this becomes a summary
(feature vector), and a head (MLP) turns that summary into the 5 numbers of the
target:

    output = [obj_logit, cx, cy, w, h]

Important detail about the output activations:
  - obj_logit: comes out RAW (no sigmoid). The BCEWithLogits loss applies the
    sigmoid internally in a numerically stable way. At inference we apply
    torch.sigmoid() to read the "has paddle" probability.
  - cx, cy, w, h: go through a SIGMOID here, ensuring they stay in [0,1] —
    the same range as the normalized targets from preprocess.

** Why TWO different branches (GAP vs spatial flatten) **
The question "IS there a paddle?" is GLOBAL: a summary of the whole image is
enough, so the presence branch uses Global Average Pooling (GAP), which averages
the map and discards position — great for "what", terrible for "where".
The question "WHERE is the box?" on the other hand is SPATIAL: it needs to know
in which corner of the map the paddle appeared. That is why the box branch does
NOT use GAP — it flattens the feature map preserving the spatial layout. (Using
GAP here too was what, in the 1st attempt, gave IoU ~0.07: the network could
only guess the average box because the position information had been thrown away.)

Deliberately small: runs comfortably on CPU/MPS on the Mac and exports easily to
ONNX -> (later) NPU. No pre-trained weights (keeps the export simple and does not
depend on a download). Template inherited from the MoveNet of the Tibia project —
we swapped the classification head (9 logits) for a detection head (5 outputs).
"""

import torch
import torch.nn as nn

IMG_SIZE = 320


def _conv_block(cin, cout):
    """conv 3x3 (keeps size) + BatchNorm + ReLU + maxpool (halves it)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class PaddleDetNet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128, 128, 128)):
        super().__init__()
        # Encoder: 6 blocks. Each block halves the resolution.
        #   320 -> 160 -> 80 -> 40 -> 20 -> 10 -> 5, with 128 channels at the end.
        # Reducing down to 5x5 (and not stopping at 10x10) keeps the flattened vector
        # of the box branch small enough to train well — at 10x10 the Linear became
        # huge (25600 inputs), ill-conditioned, and the branch would not converge.
        chans = [3] + list(widths)
        self.encoder = nn.Sequential(
            *[_conv_block(chans[i], chans[i + 1]) for i in range(len(widths))]
        )
        feat = widths[-1]                    # channels at the end of the encoder (128)
        self.feat_map = IMG_SIZE // (2 ** len(widths))   # 320/64 = 5 (map side)

        # --- PRESENCE branch (global): GAP collapses the 5x5 map into a vector of 128.
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.obj_head = nn.Sequential(
            nn.Linear(feat, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),                # 1 logit: has paddle? (raw)
        )

        # --- BOX branch (spatial): flattens the 5x5x128 map PRESERVING the
        #     position, so the network knows WHERE in the map the paddle appeared.
        flat = feat * self.feat_map * self.feat_map      # 128*5*5 = 3200
        self.box_head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),               # cx, cy, w, h (via sigmoid in forward)
        )

    def forward(self, x):
        f = self.encoder(x)                  # (B, 128, 5, 5)
        obj = self.obj_head(self.gap(f).flatten(1))      # (B, 1)  raw logit
        box = torch.sigmoid(self.box_head(f.flatten(1)))  # (B, 4) in [0,1]
        return torch.cat([obj, box], dim=1)  # (B, 5) = [obj_logit, cx, cy, w, h]


if __name__ == "__main__":
    m = PaddleDetNet()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"PaddleDetNet — {n_params/1e6:.2f}M parameters")
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    out = m(x)
    print("output:", tuple(out.shape))        # (2, 5)
    print("example:", out[0].detach().numpy())
