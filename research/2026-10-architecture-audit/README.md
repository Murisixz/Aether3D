# Architecture and language audit (October 2026)

An evidence-driven audit of Aether3D's architecture and implementation languages,
made on 15 October 2026 against `main` at 38697d5 (Stages 1-10) plus the Stage 11
branch at 403581f (PR #13, since merged). It asked whether Aether3D should change
languages for runtime performance, and answered: keep Python, fix the architecture
first, then add one native language (Rust through PyO3) for a short list of scalar
kernels; treat hardware ray tracing as a separate, later decision. That answer is the
specification of the performance modernization (see
[docs/PERFORMANCE.md](../../docs/PERFORMANCE.md)).

- [REPORT.md](REPORT.md) -- the report (a Markdown export of the published page).
- [results/](results/) -- the raw measurements every table of the report comes from.
- [experiments/](experiments/) -- the scripts and kernel sources that produced them.
- [report/build_report.py](report/build_report.py) -- regenerates the report's HTML
  page from `results/` (`python report/build_report.py`).

Everything here is kept as it ran: research code, not part of the package, excluded
from the linters (`pyproject.toml`). Paths of the original machine were replaced by
paths relative to the repository; nothing else was changed. Toolchains, virtual
environments, build directories, compiled libraries and the experiments' large binary
inputs are not kept (`.gitignore`); the commands below rebuild them.

## The machine and software

- Intel Core i9-12900H (6 performance + 8 efficiency cores, 20 threads), 15 GiB;
  NVIDIA GeForce RTX 4060 Laptop GPU (Vulkan, driver 615.78.08); Arch Linux,
  kernel 7.2.9.
- CPython 3.14.8, NumPy 2.5.4, wgpu-py 0.32.0 (wgpu-native 29), Pillow 12.3.0.
- gcc 16.2.1, clang 23.1.1, Rust 1.99.1, Zig 0.17.0, Go 1.27.2, .NET 10.0.402
  (Native AOT), Numba 0.68, Cython 3.3, nanobind 3.1, PyO3 0.30.1, cffi 2.1,
  Embree 4.4.1 (release binaries), Rust `wgpu` 30.0.1.

Results depend on the machine: compare runs on one machine only. Every experiment
checks its outputs against Aether3D's own (bitwise where the arithmetic is the same;
agreement rates for float32 kernels: Embree and the GPU).

## Experiments

Run the Python scripts from the repository root with `PYTHONPATH=src` and an
interpreter that has NumPy (plus Numba, Cython, cffi, nanobind or Pillow where an
experiment uses them). `E` below stands for the experiment's folder.

| Folder | Question | Results |
| --- | --- | --- |
| `subsystems` | Where does each subsystem spend its time at 10,000 objects? Which modules import which? | `subsystems.json`, `imports.json` |
| `lifecycle` | Is a scene freed after it was queried, rendered, ...? (Found the spatial index leak fixed in this branch.) | printed |
| `e1_world_matrices` | World matrices of a hierarchy: today's code vs the same algorithm on flat arrays in NumPy, Numba, Cython, C, Rust, Zig, Go, C# | `e1_world_matrices.json` |
| `e2_instance_data` | Per-frame instance records: packing, dirty updates, GPU uploads, GPU camera rebase precision | `e2_instance_data_*.json` |
| `e3_traversal` | BVH closest-hit traversal with Aether3D's exact algorithm in 8 languages, 1/8/20 threads, plus Embree | `e3_*.json` |
| `e4_boundary` | Cost of a call from Python into native code for each binding technology; object creation profile | `e4_boundary.json` |
| `e5_topology` | Mesh edge topology: `EditableMesh.from_mesh` vs NumPy, C, Rust (same edge ids) | `e5_topology.json` |
| `e6_png` | PNG decode/encode: Aether3D vs Pillow vs zlib + a C unfilter | `e6_png.json` |
| `e7_gpu_ray_queries` | GPU traversal: Aether3D's WGSL `trace_ray` vs hardware ray queries (Rust wgpu) | `e7_runs.jsonl`, `e7_agreement.json` |

### Building the kernels

```sh
# E1 (in E = experiments/e1_world_matrices)
gcc -O3 -march=native -ffp-contract=off -shared -fPIC -o libworld_c.so world.c
gcc -O3 -march=native -ffp-contract=fast -shared -fPIC -o libworld_c_fma.so world.c
clang -O3 -march=native -ffp-contract=off -shared -fPIC -o libworld_clang.so world.c
python setup_cy.py build_ext --inplace                      # Cython
(cd rust && RUSTFLAGS="-C target-cpu=native" cargo build --release && cp target/release/libworld_rs.so ..)
zig build-lib -dynamic -OReleaseFast -mcpu=native -femit-bin=libworld_zig.so world.zig
(cd go && CGO_ENABLED=1 go build -buildmode=c-shared -o ../libworld_go.so world.go)
(cd csharp && dotnet publish -c Release -r linux-x64 -o out && cp out/world_cs.so ../world_cs4.so)
python E/bench_e1.py E            # and: python E/aether_world.py E

# E2
gcc -O3 -march=native -ffp-contract=off -shared -fPIC -o libpack.so pack.c
python E/bench_e2_cpu.py E;  python E/bench_e2_gpu_precision.py

# E3
gcc -O3 -march=native -ffp-contract=off -fopenmp -shared -fPIC -o libtrace_c_omp.so trace.c -lm
clang -O3 -march=native -ffp-contract=off -shared -fPIC -o libtrace_clang.so trace.c -lm
g++ -std=c++20 -O3 -march=native -ffp-contract=off -fopenmp -shared -fPIC -o libtrace_cpp.so trace_cpp.cpp
(cd rust && RUSTFLAGS="-C target-cpu=native" cargo build --release && cp target/release/libtrace_rs.so ..)
zig build-lib -dynamic -OReleaseFast -mcpu=native -lc -femit-bin=libtrace_zig.so trace.zig
(cd go && CGO_ENABLED=1 go build -buildmode=c-shared -o ../libtrace_go.so .)
(cd csharp && dotnet publish -c Release -r linux-x64 -o out && cp out/trace_cs.so ../libtrace_cs.so)
# Embree 4.4.1: unpack embree-4.4.1.x86_64.linux.tar.gz into $EMBREE, then
gcc -O3 -march=native -fopenmp -shared -fPIC -I$EMBREE/include -o libembree_bench.so \
    embree_bench.c -L$EMBREE/lib64 -lembree4 -Wl,-rpath,$EMBREE/lib64 -lm
python E/bench_e3.py E; python E/bench_mt_interleaved.py E; python E/bench_cpp.py E
python E/bench_embree.py E; python E/numba_latency.py E

# E4 (EXT = python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
gcc -O3 -shared -fPIC -I"$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
    capi.c -o capi_mod$EXT
gcc -O3 -shared -fPIC ffi_c.c -o libffi_c.so
python build_cffi.py; cythonize -3 -i cy_mod.pyx
cmake -S . -B nbbuild -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build nbbuild && cp nbbuild/nb_mod*.so .
(cd pyo3 && PYO3_PYTHON=python cargo build --release && cp target/release/libpyo3_mod.so ../pyo3_mod$EXT)
python -I E/bench_ffi.py E;  python E/prof_create.py

# E5
gcc -O3 -march=native -shared -fPIC -o libtopo_c.so topo.c
(cd rust && RUSTFLAGS="-C target-cpu=native" cargo build --release && cp target/release/libtopo_rs.so ..)
python E/bench_e5.py E;  python E/prof_from_mesh.py

# E6
gcc -O3 -march=native -shared -fPIC -o libunfilter.so unfilter.c
python E/bench_e6.py E

# E7 (needs E3's libtrace_c_omp.so; writes about 300 MB of rays and BVH words to E/data)
python E/export.py
(cd rtbench && cargo build --release)
for s in sphere100k cubes400 sphere1m; do rtbench/target/release/rtbench E/data $s 10; done > runs.jsonl
python E/check.py E/data
```

Notes from the runs:

- **FMA contraction.** GCC's default for GNU C contracts `a*b+c` into fused
  multiply-adds when the target has them (`-march=native` here). The parity builds
  use `-ffp-contract=off`; the FMA build of E1 differs from the reference in the 13th
  digit.
- **C# Native AOT.** `stackalloc` in an `[UnmanagedCallersOnly]` export aborted on
  entry with `x86-64-v3`/`native` instruction sets; the archived E1 variant uses a
  heap array.
- **Laptop variance.** Multi-threaded runs drift with heat and turbo state (the same
  20-thread C kernel measured 43 and 60 M rays/s in two runs);
  `bench_mt_interleaved.py` interleaves implementations in random order over 7 rounds
  and is the run the report uses.
