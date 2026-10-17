// Aether3D GPU baking rays (aether3d/texturing/gpu_bake.py): the CPU path tracer's
// two-level BVH and watertight triangle test (traverse.wgsl), applied to bake rays.
//
// Modes (BakeParams.mode), one invocation per item:
// * 0, closest: an item is a ray (origin, tmax, direction, skipped instance and triangle);
//   the output is its closest hit (t, instance, triangle, barycentrics).
// * 1, occlusion: an item is a surface point (origin, normal, Cranley-Patterson shift,
//   skipped instance and triangle); BakeParams.rays cosine-weighted directions around the
//   normal (a Hammersley set rotated by the shift: texturing/bake.py's hemisphere_rays) are
//   traced as any-hit rays up to BakeParams.distance; the output is the visible fraction.
// * 2, distance: as 1, with closest hits; the output is the mean distance to the next
//   surface (misses count as BakeParams.distance), divided by BakeParams.distance.

struct BakeParams {
  root_lo: vec3<f32>,
  instance_count: u32,
  root_hi: vec3<f32>,
  mode: u32,
  count: u32,
  rays: u32,
  distance: f32,
  row: u32,  // items per dispatch row (2D dispatch)
};

@group(0) @binding(0) var<storage, read> pt: array<u32>;
@group(0) @binding(1) var<uniform> bp: BakeParams;
@group(0) @binding(2) var<storage, read> items: array<f32>;      // 12 floats per item
@group(0) @binding(3) var<storage, read_write> results: array<f32>;  // 6 floats per item

fn trace_root_lo() -> vec3<f32> {
  return bp.root_lo;
}

fn trace_root_hi() -> vec3<f32> {
  return bp.root_hi;
}

fn trace_instance_count() -> u32 {
  return bp.instance_count;
}

const FRAME_TOLERANCE_BAKE: f32 = 1e-5;
const TWO_PI: f32 = 6.283185307179586;

fn radical_inverse(i_in: u32) -> f32 {
  var x = i_in;
  x = (x << 16u) | (x >> 16u);
  x = ((x & 0x55555555u) << 1u) | ((x & 0xAAAAAAAAu) >> 1u);
  x = ((x & 0x33333333u) << 2u) | ((x & 0xCCCCCCCCu) >> 2u);
  x = ((x & 0x0F0F0F0Fu) << 4u) | ((x & 0xF0F0F0F0u) >> 4u);
  x = ((x & 0x00FF00FFu) << 8u) | ((x & 0xFF00FF00u) >> 8u);
  return f32(x) * (1.0 / 4294967296.0);
}

// pathtrace/sampling.py's frame(): t, b with (t, b, n) right-handed and orthonormal.
fn bake_direction(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
  let sign = select(-1.0, 1.0, n.z >= -FRAME_TOLERANCE_BAKE);
  let a = -1.0 / (sign + n.z);
  let b = n.x * n.y * a;
  let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
  let bt = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
  let r = sqrt(u1);
  let phi = TWO_PI * u2;
  let z = sqrt(max(0.0, 1.0 - u1));
  return t * (r * cos(phi)) + bt * (r * sin(phi)) + n * z;
}

@compute @workgroup_size(64)
fn bake_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * bp.row + gid.x;
  if (gid.x >= bp.row || i >= bp.count) {
    return;
  }
  let base = 12u * i;
  let o = vec3<f32>(items[base], items[base + 1u], items[base + 2u]);
  let skip_inst = bitcast<u32>(items[base + 10u]);
  let skip_prim = bitcast<u32>(items[base + 11u]);
  let out = 6u * i;
  if (bp.mode == 0u) {
    let tmax = items[base + 3u];
    let d = vec3<f32>(items[base + 4u], items[base + 5u], items[base + 6u]);
    let h = trace_ray(o, d, 0.0, tmax, skip_inst, skip_prim, false);
    results[out] = h.t;
    results[out + 1u] = bitcast<f32>(h.inst);
    results[out + 2u] = bitcast<f32>(h.prim);
    results[out + 3u] = h.b.x;
    results[out + 4u] = h.b.y;
    results[out + 5u] = h.b.z;
    return;
  }
  let n = vec3<f32>(items[base + 4u], items[base + 5u], items[base + 6u]);
  let s1 = items[base + 7u];
  let s2 = items[base + 8u];
  let count = bp.rays;
  var acc = 0.0;
  for (var r = 0u; r < count; r++) {
    let u1 = fract((f32(r) + 0.5) / f32(count) + s1);
    let u2 = fract(radical_inverse(r) + s2);
    let d = bake_direction(n, u1, u2);
    if (bp.mode == 1u) {
      let h = trace_ray(o, d, 0.0, bp.distance, skip_inst, skip_prim, true);
      acc += select(1.0, 0.0, h.inst != PT_NONE);
    } else {
      let h = trace_ray(o, d, 0.0, bp.distance, skip_inst, skip_prim, false);
      acc += select(bp.distance, min(h.t, bp.distance), h.inst != PT_NONE);
    }
  }
  if (bp.mode == 1u) {
    results[out] = acc / f32(count);
  } else {
    results[out] = acc / (f32(count) * bp.distance);
  }
}
