"""E2 (GPU): upload strategies and a GPU-side camera rebase, on the RTX 4060 through wgpu-py."""
import json, statistics, time
import numpy as np
import wgpu

adapter = [a for a in wgpu.gpu.enumerate_adapters_sync() if "NVIDIA" in a.info["device"] and a.info["backend_type"] == "Vulkan"][0]
device = adapter.request_device_sync(required_features=["timestamp-query"])
q = device.queue
N, REC = 10_000, 128
data = np.random.default_rng(0).random((N, 32)).astype(np.float32)
U = wgpu.BufferUsage
buf = device.create_buffer(size=N * REC, usage=U.STORAGE | U.COPY_DST | U.COPY_SRC)

tiny = device.create_buffer(size=16, usage=U.COPY_SRC | U.COPY_DST)
def wait():
    q.submit([device.create_command_encoder().finish()])
    q.read_buffer(tiny)  # waits for the queue (on_submitted_work_done_sync is broken in wgpu-py 0.32)

def med(fn, k=30, pre=5):
    for _ in range(pre): fn()
    ts = []
    for _ in range(k):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

out = {}
out["submit_and_wait_roundtrip_us"] = med(wait) * 1e6
def new_buffer():
    b = device.create_buffer_with_data(data=data, usage=U.STORAGE); b.destroy()
out["create_buffer_with_data_1.28MB_us"] = med(new_buffer) * 1e6
out["write_buffer_full_1.28MB_us"] = med(lambda: q.write_buffer(buf, 0, data)) * 1e6
for k in (1, 10, 100):
    rows = np.sort(np.random.default_rng(k).choice(N, k, replace=False))
    def separate():
        for r in rows:
            q.write_buffer(buf, int(r) * REC, data[r])
    out[f"write_buffer_{k}_records_separate_us"] = med(separate) * 1e6
    out[f"write_buffer_{k}_records_one_range_us"] = med(lambda: q.write_buffer(buf, 0, data[:k])) * 1e6
wait()
# GPU-side camera rebase: world translation as float32 hi/lo pairs -> camera-relative float32, for all instances
shader = """
struct Params { eye_hi: vec4<f32>, eye_lo: vec4<f32>, count: u32 };
@group(0) @binding(0) var<storage, read> hi: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> lo: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> inst: array<array<f32, 32>>;
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.count) { return; }
    let rel = (hi[i].xyz - p.eye_hi.xyz) + (lo[i].xyz - p.eye_lo.xyz);
    inst[i][3] = rel.x; inst[i][7] = rel.y; inst[i][11] = rel.z;
}
"""
world_t = 1.0e5 + np.random.default_rng(1).uniform(-20.0, 20.0, (N, 3))  # far from the origin, near the camera
hi = np.zeros((N, 4), np.float32); hi[:, :3] = world_t.astype(np.float32)
lo = np.zeros((N, 4), np.float32); lo[:, :3] = (world_t - hi[:, :3].astype(np.float64)).astype(np.float32)
eye = np.array([100003.7071, 99998.1234, 100001.5555])
eh = np.zeros(4, np.float32); eh[:3] = eye.astype(np.float32)
el = np.zeros(4, np.float32); el[:3] = (eye - eh[:3].astype(np.float64)).astype(np.float32)
params = np.zeros(12, np.float32); params[0:4] = eh; params[4:8] = el; params.view(np.uint32)[8] = N
hb = device.create_buffer_with_data(data=hi, usage=U.STORAGE)
lb = device.create_buffer_with_data(data=lo, usage=U.STORAGE)
pb = device.create_buffer_with_data(data=params, usage=U.UNIFORM | U.COPY_DST)
module = device.create_shader_module(code=shader)
pipe = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
bg = device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
    {"binding": 0, "resource": {"buffer": hb}}, {"binding": 1, "resource": {"buffer": lb}},
    {"binding": 2, "resource": {"buffer": buf}}, {"binding": 3, "resource": {"buffer": pb}}])
qs = device.create_query_set(type="timestamp", count=2)
qbuf = device.create_buffer(size=16, usage=U.QUERY_RESOLVE | U.COPY_SRC)
rbuf = device.create_buffer(size=16, usage=U.COPY_DST | U.MAP_READ)
def rebase(timed=False):
    q.write_buffer(pb, 0, params)
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass(timestamp_writes={"query_set": qs, "beginning_of_pass_write_index": 0, "end_of_pass_write_index": 1} if timed else None)
    cp.set_pipeline(pipe); cp.set_bind_group(0, bg); cp.dispatch_workgroups((N + 63) // 64); cp.end()
    if timed:
        enc.resolve_query_set(qs, 0, 2, qbuf, 0); enc.copy_buffer_to_buffer(qbuf, 0, rbuf, 0, 16)
    q.submit([enc.finish()]); q.read_buffer(tiny)
out["gpu_rebase_10k_submit_and_wait_us"] = med(rebase) * 1e6
gpu_ns = []
for _ in range(20):
    rebase(True); rbuf.map_sync(wgpu.MapMode.READ); t = np.frombuffer(rbuf.read_mapped(), dtype=np.uint64).copy(); rbuf.unmap()
    gpu_ns.append(int(t[1] - t[0]))
out["gpu_rebase_10k_kernel_us (timestamps)"] = statistics.median(gpu_ns) / 1e3
# correctness: compare with float64 reference
rd = device.queue.read_buffer(buf)
inst = np.frombuffer(rd, dtype=np.float32).reshape(N, 32)
ref = (world_t - eye).astype(np.float32)
err = np.max(np.abs(inst[:, [3, 7, 11]].astype(np.float64) - (world_t - eye)))
out["gpu hi/lo rebase: max abs error vs float64 (m)"] = float(err)
out["current CPU path (f64 subtract, then round to f32): max abs error (m)"] = float(np.max(np.abs(ref.astype(np.float64) - (world_t - eye))))
out["naive float32 world - eye: max abs error (m)"] = float(np.max(np.abs((world_t.astype(np.float32) - eye.astype(np.float32)).astype(np.float64) - (world_t - eye))))
print(json.dumps(out, indent=1))
