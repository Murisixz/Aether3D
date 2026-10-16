"""E6: PNG decode/encode: Aether3D (NumPy) vs Pillow 12 (libpng-style C) vs zlib + C unfilter; real renders from the repo."""
import ctypes, io, json, statistics, sys, time, zlib
from pathlib import Path
import numpy as np
from PIL import Image
from aether3d.materials.images import decode_png
from aether3d.rendering.png import encode_png
E = sys.argv[1]
lib = ctypes.CDLL(f"{E}/libunfilter.so"); lib.unfilter.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 2; lib.unfilter.restype = ctypes.c_int
def med(fn, k=3):
    fn(); ts = []
    for _ in range(k):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), r
out = {}
for name in ("castle_path.png", "hero_gpu.png"):
    data = open(Path(__file__).resolve().parents[4] / "examples" / "output" / name, "rb").read()
    pil = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA" if Image.open(io.BytesIO(data)).mode == "RGBA" else "RGB"))
    h, w, ch = pil.shape; mp = h * w / 1e6
    row = {"size": f"{w}x{h}x{ch}", "bytes": len(data)}
    t, img = med(lambda: decode_png(data))
    pixels = np.asarray(img.pixels if hasattr(img, "pixels") else img.data)
    row["Aether3D decode_png ms"] = round(t * 1e3, 1); row["Aether3D decode MP/s"] = round(mp / t, 2)
    t, _ = med(lambda: np.asarray(Image.open(io.BytesIO(data)).convert(Image.open(io.BytesIO(data)).mode)), 5)
    row["Pillow decode ms"] = round(t * 1e3, 1); row["Pillow decode MP/s"] = round(mp / t, 1)
    # split: inflate (zlib, C) + unfilter (C) on the same IDAT stream as the file
    import struct
    pos, idat = 8, b""
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]; kind = data[pos + 4:pos + 8]
        if kind == b"IDAT": idat += data[pos + 8:pos + 8 + n]
        pos += 12 + n
    t_inf, raw = med(lambda: zlib.decompress(idat), 5)
    rawa = np.frombuffer(raw, np.uint8); stride = w * ch; dst = np.empty(h * stride, np.uint8)
    t_unf, rc = med(lambda: lib.unfilter(h, stride, ch, rawa.ctypes.data, dst.ctypes.data), 5)
    assert rc == 0 and np.array_equal(dst.reshape(h, w, ch), pil), "C unfilter disagrees with Pillow"
    filt = np.bincount(rawa.reshape(h, stride + 1)[:, 0], minlength=5)
    row["filter types used (None,Sub,Up,Avg,Paeth)"] = filt.tolist()
    row["zlib inflate ms"] = round(t_inf * 1e3, 1); row["C unfilter ms"] = round(t_unf * 1e3, 1)
    row["zlib + C unfilter MP/s"] = round(mp / (t_inf + t_unf), 1)
    row["Aether3D decode matches Pillow"] = bool(np.array_equal(np.asarray(pixels).reshape(pil.shape)[..., :ch], pil)) if np.asarray(pixels).size == pil.size else "shape differs"
    # encode at the same zlib level (6)
    t, enc = med(lambda: encode_png(pil))
    row["Aether3D encode_png ms"] = round(t * 1e3, 1); row["Aether3D encode bytes"] = len(enc)
    def pil_save():
        b = io.BytesIO(); Image.fromarray(pil).save(b, "PNG", compress_level=6); return b.getvalue()
    t, enc2 = med(pil_save)
    row["Pillow encode ms"] = round(t * 1e3, 1); row["Pillow encode bytes"] = len(enc2)
    out[name] = row
    print(name, json.dumps(row), flush=True)
json.dump(out, open(f"{E}/e6.json", "w"), indent=1)
