"""A single-stroke font for drawing text in raster output (no font files, no
dependencies): glyphs are polylines on a 4 x 6 grid (cap height 6, baseline 0,
descenders to -2), drawn with the same line rasterizer as everything else, so PNG
drawings are reproducible bit for bit. Lowercase letters are drawn as small capitals;
characters without a glyph are drawn as a box.

Vector outputs (SVG, DXF) write text as text, not strokes.
"""

from __future__ import annotations

import numpy as np

# Each glyph: strokes separated by ";", points "x,y" separated by spaces.
_STROKES = {
    "A": "0,0 2,6 4,0; 0.7,2 3.3,2",
    "B": "0,0 0,6 3,6 4,5 4,4 3,3 0,3; 3,3 4,2 4,1 3,0 0,0",
    "C": "4,5 3,6 1,6 0,5 0,1 1,0 3,0 4,1",
    "D": "0,0 0,6 2.5,6 4,4.5 4,1.5 2.5,0 0,0",
    "E": "4,6 0,6 0,0 4,0; 0,3 3,3",
    "F": "4,6 0,6 0,0; 0,3 3,3",
    "G": "4,5 3,6 1,6 0,5 0,1 1,0 3,0 4,1 4,3 2.5,3",
    "H": "0,0 0,6; 4,0 4,6; 0,3 4,3",
    "I": "1,6 3,6; 2,6 2,0; 1,0 3,0",
    "J": "4,6 4,1 3,0 1,0 0,1",
    "K": "0,0 0,6; 4,6 0,2; 1.3,3.3 4,0",
    "L": "0,6 0,0 4,0",
    "M": "0,0 0,6 2,3 4,6 4,0",
    "N": "0,0 0,6 4,0 4,6",
    "O": "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0",
    "P": "0,0 0,6 3,6 4,5 4,4 3,3 0,3",
    "Q": "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0; 2.5,1.5 4,0",
    "R": "0,0 0,6 3,6 4,5 4,4 3,3 0,3; 2,3 4,0",
    "S": "4,5 3,6 1,6 0,5 0,4 1,3 3,3 4,2 4,1 3,0 1,0 0,1",
    "T": "0,6 4,6; 2,6 2,0",
    "U": "0,6 0,1 1,0 3,0 4,1 4,6",
    "V": "0,6 2,0 4,6",
    "W": "0,6 1,0 2,3 3,0 4,6",
    "X": "0,0 4,6; 0,6 4,0",
    "Y": "0,6 2,3 4,6; 2,3 2,0",
    "Z": "0,6 4,6 0,0 4,0",
    "0": "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0; 0.5,1 3.5,5",
    "1": "1,5 2,6 2,0; 1,0 3,0",
    "2": "0,5 1,6 3,6 4,5 4,4 0,0 4,0",
    "3": "0,5 1,6 3,6 4,5 4,4 3,3 2,3; 3,3 4,2 4,1 3,0 1,0 0,1",
    "4": "3,0 3,6 0,2 4,2",
    "5": "4,6 0,6 0,3 3,3 4,2 4,1 3,0 1,0 0,1",
    "6": "3,6 1,6 0,5 0,1 1,0 3,0 4,1 4,2 3,3 0,3",
    "7": "0,6 4,6 1.5,0",
    "8": "1,3 0,4 0,5 1,6 3,6 4,5 4,4 3,3 1,3 0,2 0,1 1,0 3,0 4,1 4,2 3,3",
    "9": "4,3 1,3 0,4 0,5 1,6 3,6 4,5 4,1 3,0 1,0",
    ".": "1.75,0 2.25,0 2.25,0.5 1.75,0.5 1.75,0",
    ",": "2.25,0.5 2.25,0 1.5,-1",
    "-": "0.5,3 3.5,3",
    "+": "0.5,3 3.5,3; 2,1.5 2,4.5",
    "=": "0.5,2 3.5,2; 0.5,4 3.5,4",
    ":": "2,1 2,1.5; 2,4 2,4.5",
    ";": "2,4 2,4.5; 2,1.5 2,1 1.5,0",
    "/": "0,0 4,6",
    "\\": "0,6 4,0",
    "(": "3,6 2,5 1.5,3 2,1 3,0",
    ")": "1,6 2,5 2.5,3 2,1 1,0",
    "[": "3,6 1.5,6 1.5,0 3,0",
    "]": "1,6 2.5,6 2.5,0 1,0",
    "'": "2,6 2,4.5",
    '"': "1.25,6 1.25,4.5; 2.75,6 2.75,4.5",
    "°": "1.5,6 2.5,6 2.5,5 1.5,5 1.5,6",
    "²": "1,5.5 1.5,6 2.5,6 3,5.5 1,4 3,4",
    "³": "1,6 3,6 2,5 3,4.5 2.5,4 1,4",
    "×": "1,1.5 3,4.5; 1,4.5 3,1.5",
    "x": "1,1 3,4; 1,4 3,1",
    "_": "0,-0.5 4,-0.5",
    "#": "1,0 1.5,6; 2.5,0 3,6; 0,2 4,2; 0,4 4,4",
    "%": "0,0 4,6; 0.5,5.5 1,5.5 1,5 0.5,5 0.5,5.5; 3,1 3.5,1 3.5,0.5 3,0.5 3,1",
    "&": "4,0 1,4 1,5 1.5,6 2.5,6 3,5 3,4.5 0,2 0,1 1,0 2,0 4,2.5",
    "?": "0,5 1,6 3,6 4,5 4,4 2,3 2,2; 2,0.5 2,0",
    "!": "2,6 2,2; 2,0.5 2,0",
    "<": "4,6 0,3 4,0",
    ">": "0,6 4,3 0,0",
    "*": "2,1.5 2,4.5; 0.7,2.2 3.3,3.8; 0.7,3.8 3.3,2.2",
    "@": "3,2 3,4 1,4 1,2 3,2 4,3 4,5 3,6 1,6 0,5 0,1 1,0 3,0",
    "~": "0,3 1,3.5 3,2.5 4,3",
    "±": "0.5,3.5 3.5,3.5; 2,2 2,5; 0.5,0.5 3.5,0.5",
    "Ø": "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0; 0,-0.5 4,6.5",
    "⌀": "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0; 0,-0.5 4,6.5",
}


def _parse(spec: str) -> list[np.ndarray]:
    out = []
    for stroke in spec.split(";"):
        pts = [tuple(float(v) for v in p.split(",")) for p in stroke.split()]
        out.append(np.array(pts, dtype=np.float64))
    return out


GLYPHS = {ch: _parse(spec) for ch, spec in _STROKES.items()}
_BOX = _parse("0,0 4,0 4,6 0,6 0,0")
ADVANCE = 5.5  # grid units per character (4 wide + spacing)
CAP = 6.0


def text_strokes(
    text: str, origin, height: float, angle: float = 0.0, align: str = "left"
) -> list[np.ndarray]:
    """Polylines drawing ``text`` with cap height ``height`` from ``origin`` (the baseline
    start; ``align`` "center" or "right" shifts it), rotated by ``angle`` radians."""
    s = height / CAP
    width = (len(text) * ADVANCE - 1.5) * s
    shift = {"left": 0.0, "center": -width / 2, "right": -width}.get(align, 0.0)
    c, sn = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -sn], [sn, c]])
    out = []
    x = 0.0
    for ch in text:
        if ch == " ":
            x += ADVANCE
            continue
        small = ch.islower() and ch.upper() in GLYPHS and ch not in GLYPHS
        glyph = GLYPHS.get(ch) or GLYPHS.get(ch.upper()) or _BOX
        k = 0.75 if small else 1.0
        for stroke in glyph:
            local = (stroke * k + np.array([x + (1 - k) * 2, 0.0])) * s + np.array([shift, 0.0])
            out.append(local @ rot.T + np.asarray(origin, dtype=np.float64))
        x += ADVANCE
    return out


def text_width(text: str, height: float) -> float:
    return (len(text) * ADVANCE - 1.5) * height / CAP
