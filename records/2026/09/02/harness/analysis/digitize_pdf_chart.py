#!/usr/bin/env python3
"""Digitize curves 1a/1b off the PDF's page-1 chart.

Why: the printed labels are ambiguous.  At x=1500 three labels stack up --
885142, 831574, 761774 -- and 885142 carries the ORANGE SQUARE marker, i.e. it
belongs to curve "2. GPU + LMCache-CPU", not to 1b.  Reading 1b as 885142 (as
this repro did at first) inflates VAST's finding-(1) gap from 1.09x to 1.16x.
Pixel tracing settles it independently of the labels.

Calibration anchors, both at x=1500 and both unambiguous:
    red circle   761774   (curve 1a)
    orange square 885142  (curve 2)
x anchors: the two red circle markers wide enough to detect, c=750 and c=1500.

    uv run --with pillow --with numpy analysis/digitize_pdf_chart.py <page1.png>
"""
import sys
import numpy as np
from PIL import Image

a = np.asarray(Image.open(sys.argv[1] if len(sys.argv) > 1
                          else "analysis/pdf_page1_chart.png").convert("RGB")).astype(int)
H, W, _ = a.shape
r, g, b = a[..., 0], a[..., 1], a[..., 2]
inside = np.zeros((H, W), bool)
inside[52:840, 203:1037] = True                      # plot frame
pink = ((r > 235) & (g > 150) & (g < 215) & (abs(g - b) < 25) & ((r - g) > 45)) & inside
red = ((r > 150) & (g < 80) & (b < 80)) & inside

V = lambda y: -2108.9 * y + 1718140.0                # px -> P99 TTFT (ms)
X = lambda c: 240.5 + 0.506 * c                      # concurrency -> px


def runs(mask, x):
    ys = np.nonzero(mask[:, x])[0]
    if not len(ys):
        return []
    out, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= 2:
            cur.append(y)
        else:
            out.append((cur[0] + cur[-1]) / 2)
            cur = [y]
    out.append((cur[0] + cur[-1]) / 2)
    return out


def read(mask, c, other=None):
    """Topmost run near column X(c).  `other` rejects a neighbouring curve's
    antialiasing halo -- the pink mask picks up a fringe around the red marker."""
    x = int(round(X(c)))
    for dx in (0, -1, 1, -2, 2, -3, 3, -4, 4):
        cand = runs(mask, x + dx)
        if other is not None:
            o = runs(other, x + dx)
            cand = [q for q in cand if all(abs(q - oo) > 3 for oo in o)]
        if cand:
            return min(cand)
    return None


print(f"{'conc':>6} {'1a (red)':>11} {'1b (pink)':>11} {'1b/1a':>8}")
for c in (500, 700, 750, 800, 900, 1000, 1200, 1400):
    yr, yp = read(red, c), read(pink, c, other=red)
    if yr is None or yp is None:
        print(f"{c:>6}   (occluded by a label box)")
        continue
    vr, vp = V(yr), V(yp)
    print(f"{c:>6} {vr:>11,.0f} {vp:>11,.0f} {vp / vr:>7.3f}x")
print(f"{1500:>6} {761774:>11,} {831574:>11,} {831574 / 761774:>7.3f}x   <- printed labels")
