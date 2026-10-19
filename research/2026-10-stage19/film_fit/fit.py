"""Fit Aether3D's "film" view transform to Blender 5.2's AgX, measured (E3b).

The transform is Aether3D's own construction: a 3 x 3 gamut matrix (rows summing to 1, so
grey stays grey), a per-channel tone curve on log2 exposure (a monotone curve through the
measured grey ramp), a second 3 x 3 matrix in display-linear light, and the sRGB encoding.
This script fits the two matrices (12 numbers) to the seven measured colour ramps by
Nelder-Mead (no SciPy) and prints them; aether3d/rendering/tonemap.py carries the result.

    .venv/bin/python research/2026-10-stage19/film_fit/fit.py
"""

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.load(gzip.open(ROOT / "blender" / "e3_lights" / "blender_view_520.json.gz"))
STOPS = np.array(DATA["stops"])
AGX = DATA["transforms"]["AgX|None"]
LUM = np.array([0.2126, 0.7152, 0.0722])


def srgb_encode(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


# The grey curve: display code values against log2(exposure / 0.18), from the measurement.
GREY_EV = STOPS
GREY_D = srgb_encode(np.array(AGX["grey"])[:, 1])


def curve(ev):
    """Monotone (piecewise-linear in ev, measured points) display value of grey."""
    return np.interp(ev, GREY_EV, GREY_D, left=0.0, right=1.0)


def transform(rgb, m_in, m_out):
    x = rgb @ m_in.T
    ev = np.log2(np.maximum(x, 1e-12) / 0.18)
    y = srgb_decode(curve(ev))
    out = np.clip(y @ m_out.T, 0.0, 1.0)
    return srgb_encode(out)


def matrices(p):
    def m(a):
        a = np.asarray(a).reshape(3, 2)
        return np.column_stack([a[:, 0], a[:, 1], 1.0 - a[:, 0] - a[:, 1]])

    return m(p[:6]), m(p[6:])


SAMPLES = []
for hue, rgb in DATA["hues"].items():
    if hue == "grey":
        continue
    target = srgb_encode(np.array(AGX[hue]))
    for i, ev in enumerate(STOPS):
        if -8 <= ev <= 10:
            SAMPLES.append((np.array(rgb, float) * 0.18 * 2.0**ev, target[i]))
X = np.array([s[0] for s in SAMPLES])
Y = np.array([s[1] for s in SAMPLES])


def loss(p):
    m_in, m_out = matrices(p)
    return float(np.mean((transform(X, m_in, m_out) - Y) ** 2))


def nelder_mead(f, x0, step=0.05, iters=6000, tol=1e-12):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        x = np.array(x0, float)
        x[i] += step
        pts.append(x)
    vals = [f(x) for x in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[-1] - vals[0] < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (x - pts[0]) for x in pts]
                vals = [f(x) for x in pts]
    return pts[int(np.argmin(vals))], float(min(vals))


def start(s):
    m = (1 - s) * np.eye(3) + s * np.outer(np.ones(3), LUM)
    return m[:, :2].reshape(-1)


if __name__ == "__main__":
    p0 = np.concatenate([start(0.2), np.linalg.inv((1 - 0.2) * np.eye(3) + 0.2 *
                                                    np.outer(np.ones(3), LUM))[:, :2].reshape(-1)])
    best, val = None, np.inf
    for restart in range(6):
        p, v = nelder_mead(loss, p0 if best is None else best, step=0.08 / (restart + 1))
        if v < val:
            best, val = p, v
    m_in, m_out = matrices(best)
    err = np.abs(transform(X, m_in, m_out) - Y)
    print("rms display error", np.sqrt(val), "max", err.max(), "p95", np.percentile(err, 95))
    np.set_printoptions(precision=8, suppress=True)
    print("FILM_INSET =", repr(m_in))
    print("FILM_OUTSET =", repr(m_out))
    print("GREY curve: stops", GREY_EV.tolist())
    print("GREY_D =", [round(float(v), 6) for v in GREY_D])
