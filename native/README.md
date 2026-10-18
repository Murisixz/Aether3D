# aether3d-native

Native kernels for [Aether3D](https://github.com/Murisixz/Aether3D), in Rust through
PyO3: ray traversal (closest hit and any hit), transform hierarchies (local and world
matrices; also skeletons' forward kinematics), PNG row filters (encoding and decoding),
sparse LDL^T, atlas packing, polygon triangulation (ear clipping of many polygons with
holes per call, for CAD faces, sections and fills), linear blend skinning (deformed
positions and cofactor normals of skinned meshes, threaded by vertex range) and the
hydrology sweeps of world generation (priority-flood filling, drainage accumulation in
integers, implicit stream-power incision, sediment transport and thermal relaxation;
since 0.18.0).

Aether3D does not need it: every kernel has a NumPy implementation, which Aether3D uses
when this package is missing, cannot be loaded, or is switched off with
`AETHER3D_NATIVE=0`. With it, the same results come faster: each kernel performs the
same floating-point operations in the same order as the NumPy code (Rust never fuses
`a * b + c` into a multiply-add on its own), so results are bitwise identical, which
Aether3D's tests check (`tests/test_native.py`).

## Building

A Rust toolchain (1.85 or later) and [maturin](https://www.maturin.rs/):

```
pip install maturin
cd native
maturin build --release      # a wheel in target/wheels
pip install target/wheels/aether3d_native-*.whl
```

or, for development, `maturin develop --release` inside the environment Aether3D is
installed in. Wheels use the stable ABI of CPython 3.11 (one wheel per platform for every
CPython from 3.11); on Python 3.10, Aether3D runs on NumPy alone.

## Safety

Every function checks its arrays -- element types, contiguity, sizes, every index the
kernel follows, outputs that would share memory with another array -- and raises
`ValueError` instead of reading or writing out of bounds. The kernels release the GIL.

## License

Apache-2.0, like Aether3D. Dependencies: PyO3 (MIT OR Apache-2.0).
