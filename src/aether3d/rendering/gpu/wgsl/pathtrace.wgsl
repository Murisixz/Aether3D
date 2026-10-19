// Aether3D GPU path tracer -- the compute kernel behind backend="gpu_path".
//
// This is the CPU path tracer (aether3d/rendering/pathtrace/) ported to a GPU compute
// shader: the same estimator, the same random numbers and the same data.
//
// * Random numbers: SplitMix64 of (seed, pixel, sample, dimension), computed exactly in
//   32-bit integer pairs, so every decision draws the CPU's random number (truncated to
//   the 24 bits a float32 holds). With identical scenes and settings the two backends
//   trace the same paths; their images differ by float32 rounding and the rare decision
//   it flips.
// * Geometry: the CPU's two-level BVH (rendering/raytrace/accel.py), uploaded as flat
//   arrays and traversed with the same nearest-child-first stack machine and watertight
//   ray/triangle tests (Woop, Benthin and Wald 2013), in float32 and camera-relative
//   coordinates (the camera's eye at the origin, as on the GPU rasterizer).
// * Materials: material.wgsl's evaluate_surface -- the GPU rasterizer's own material code,
//   which agrees with the CPU's -- with texture footprints from ray differentials (camera
//   rays) and ray cones (later rays), as on the CPU.
// * Lights and the estimator: next-event estimation of every analytic light and of
//   emissive triangles (power-sampled), multiple importance sampling with the power
//   heuristic, sized lights found by BSDF rays too, the environment, Russian roulette,
//   the clamp, pass-through surfaces (alpha cut-outs, backs of single-sided materials),
//   ambient occlusion and the pixel-center pass. Constants and random-number slots match
//   the CPU's.
//
// Every invocation traces whole paths for one pixel and adds them to that pixel's sums,
// in sample order (no atomics on pixels): images repeat bit for bit on a given device.
//
// The texture-program interpreter (run_program, inside evaluate_surface) must be compiled
// once: D3D12's FXC inlines every call. Each path is therefore a small state machine whose
// loop evaluates materials at exactly one place (see run_state below).

const LIGHT_WORDS: u32 = 44u;
const EMITTER_WORDS: u32 = 12u;

// The CPU path tracer's constants (pathtrace/*.py).
const MIN_COS: f32 = 1e-6;
const RR_MIN: f32 = 0.05;
const DIFFUSE_SPREAD: f32 = 1.0;
const LOBE_MIN: f32 = 0.1;
const LOBE_MAX: f32 = 0.9;
const MAX_LAYERS: u32 = 256u;
const MIN_LIGHT_DISTANCE: f32 = 1e-4;
// Float32 stand-ins for the CPU's float64 margins: shadow rays stop this fraction short of
// a sampled point on an emitter, and normals with z down to -FRAME_TOLERANCE take the +z
// branch of the frame construction (on the CPU: -1e-12; a float32 normal of a surface that
// is vertical up to rounding has a z of either sign around 1e-8).
const SHADOW_END: f32 = 0.9999;
const FRAME_TOLERANCE: f32 = 1e-5;
const OFFSET_INT: f32 = 256.0;       // ray origins move 256 ulps off surfaces
const OFFSET_FLOAT: f32 = 1.0 / 65536.0;
const OFFSET_ORIGIN: f32 = 1.0 / 32.0;

// Random-number slots (pathtrace/sampling.py).
const SLOT_PIXEL: u32 = 0u;
const SLOT_LOBE: u32 = 2u;
const SLOT_BSDF: u32 = 3u;
const SLOT_RR: u32 = 5u;
const SLOT_LENS: u32 = 6u;
const SLOT_AO: u32 = 16u;
const SLOT_LIGHTS: u32 = 4096u;

// Params.flags and modes.
const FLAG_PERSPECTIVE: u32 = 1u;
const FLAG_PASS_THROUGH: u32 = 2u;
const FLAG_AO: u32 = 4u;
const FLAG_SIZED_LIGHTS: u32 = 8u;
const MODE_PATHS: u32 = 0u;
const MODE_CENTER: u32 = 1u;
const MODE_RNG: u32 = 2u;
const INTEGRATOR_AO: u32 = 1u;
const CULL_BACK: u32 = 1u;
const CULL_FRONT: u32 = 2u;
const ENV_AMBIENT: u32 = 0u;
const ENV_SKY: u32 = 2u;
const ENV_IMAGE: u32 = 3u;
const ENV_VISIBLE: u32 = 1u;  // bits of bitcast<u32>(params.env_image.w)
const ENV_SAMPLED: u32 = 2u;

// Accumulator planes (n_pix floats each).
const ACC_DIRECT: u32 = 0u;
const ACC_INDIRECT: u32 = 3u;
const ACC_ALBEDO: u32 = 6u;
const ACC_NORMAL: u32 = 9u;
const ACC_AO: u32 = 12u;
const ACC_HITS: u32 = 13u;
const ACC_MEAN: u32 = 14u;
const ACC_M2: u32 = 17u;
const ACC_DEPTH: u32 = 20u;
const ACC_INSTANCE: u32 = 21u;

// Counters.
const C_CAMERA: u32 = 0u;
const C_BOUNCE: u32 = 1u;
const C_SHADOW: u32 = 2u;
const C_AO: u32 = 3u;
const C_CENTER: u32 = 4u;
const C_THROUGH: u32 = 5u;
const C_LIMITED: u32 = 6u;
const C_VISIBLE: u32 = 8u;

// A path's ray counts. Scalar fields, not an array: D3D12's FXC cannot write an array
// inside a struct ("l-value expected"). Counters are indexed by constants only.
struct Tally {
  camera: u32,
  bounce: u32,
  shadow: u32,
  ao: u32,
  center: u32,
  through: u32,
  limited: u32,
};

fn add_tally(a: Tally, b: Tally) -> Tally {
  return Tally(
    a.camera + b.camera,
    a.bounce + b.bounce,
    a.shadow + b.shadow,
    a.ao + b.ao,
    a.center + b.center,
    a.through + b.through,
    a.limited + b.limited,
  );
}

struct Params {
  eye: vec3<f32>, near: f32,             // eye: world position of the camera (for world coordinates)
  forward: vec3<f32>, far: f32,
  right: vec3<f32>, hx: f32,
  up: vec3<f32>, hy: f32,
  env_uniform: vec3<f32>, env_kind: u32,
  background: vec3<f32>, light_count: u32,
  zenith: vec3<f32>, emitter_count: u32,
  horizon: vec3<f32>, instance_count: u32,
  ground: vec3<f32>, flags: u32,
  seed_lo: u32, seed_hi: u32, width: u32, height: u32,
  sample_first: u32, sample_count: u32, bounces: u32, light_samples: u32,
  rr_vertex: u32, integrator: u32, cull: u32, mode: u32,
  clamp: f32, diff_scale: f32, ao_samples: u32, ao_distance: f32,
  ao_strength: f32, stack_size: u32, tile_x: u32, tile_y: u32,
  tile_w: u32, tile_h: u32, n_pix: u32, _pad0: u32,
  root_lo: vec3<f32>, _pad1: f32,
  root_hi: vec3<f32>, _pad2: f32,
  atmo: Atmo,  // scene.world.atmosphere (atmosphere.wgsl): sky glows, fog on camera paths
  lens: vec4<f32>,      // lens shift (NDC offsets x, y), aperture radius (m; 0: pinhole), focus
  aperture: vec4<f32>,  // blades (0: round), cos and sin of the rotation, 1 / ratio
  env_image: vec4<f32>, // image environment: cos, sin of its rotation, intensity, flag bits
};

@group(0) @binding(8) var<storage, read> pt: array<u32>;
@group(0) @binding(9) var<uniform> params: Params;
@group(0) @binding(10) var<storage, read_write> accum: array<f32>;
@group(0) @binding(11) var<storage, read_write> counters: array<atomic<u32>>;

fn trace_root_lo() -> vec3<f32> {
  return params.root_lo;
}

fn trace_root_hi() -> vec3<f32> {
  return params.root_hi;
}

fn trace_instance_count() -> u32 {
  return params.instance_count;
}

var<workgroup> wg_counts: array<atomic<u32>, 8>;

fn camera_eye() -> vec3<f32> {
  return params.eye;
}

fn mul_wide(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  return vec2<u32>((p00 & 0xffffu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let low = mul_wide(a.x, b.x);
  return vec2<u32>(low.x, low.y + a.y * b.x + a.x * b.y);
}

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let lo = a.x + b.x;
  return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn shr64(a: vec2<u32>, k: u32) -> vec2<u32> {  // 0 < k < 32
  return vec2<u32>((a.x >> k) | (a.y << (32u - k)), a.y >> k);
}

fn mix64(x_in: vec2<u32>) -> vec2<u32> {
  var x = add64(x_in, GOLDEN);
  x = mul64(x ^ shr64(x, 30u), MIX_1);
  x = mul64(x ^ shr64(x, 27u), MIX_2);
  return x ^ shr64(x, 31u);
}

fn path_key(pixel: u32, sample: u32) -> vec2<u32> {
  var key = mix64(vec2<u32>(params.seed_lo, params.seed_hi));
  key = mix64(key ^ mul64(vec2<u32>(pixel, 0u), PIXEL_MUL));
  return mix64(key ^ mul64(vec2<u32>(sample, 0u), SAMPLE_MUL));
}

// The uniform number in [0, 1) of decision `slot` at path vertex `vertex`.
fn rnd(key: vec2<u32>, vertex: u32, slot: u32) -> f32 {
  let x = mix64(key ^ add64(mul64(vec2<u32>(slot, vertex), DIM_MUL), GOLDEN));
  return f32(x.y >> 8u) * (1.0 / 16777216.0);
}

// -- frames and warps (pathtrace/sampling.py) -----------------------------------------------

struct Basis {
  t: vec3<f32>,
  b: vec3<f32>,
};

fn basis(n: vec3<f32>) -> Basis {
  let sign = select(-1.0, 1.0, n.z >= -FRAME_TOLERANCE);
  let a = -1.0 / (sign + n.z);
  let b = n.x * n.y * a;
  return Basis(
    vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
    vec3<f32>(b, sign + n.y * n.y * a, -n.y)
  );
}

fn to_local(v: vec3<f32>, f: Basis, n: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(v, f.t), dot(v, f.b), dot(v, n));
}

fn to_world(v: vec3<f32>, f: Basis, n: vec3<f32>) -> vec3<f32> {
  return f.t * v.x + f.b * v.y + n * v.z;
}

fn cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
}

// Uniform in the cone z >= 1 - omc (omc = 1 - cos(half angle), exact from the host).
fn uniform_cone(u1: f32, u2: f32, omc: f32) -> vec3<f32> {
  let one_minus_z = u1 * omc;
  let r = sqrt(max(0.0, one_minus_z * (2.0 - one_minus_z)));
  let phi = 2.0 * PI * u2;
  return vec3<f32>(r * cos(phi), r * sin(phi), 1.0 - one_minus_z);
}

fn uniform_triangle(u1: f32, u2: f32) -> vec3<f32> {
  let s = sqrt(u1);
  let b0 = 1.0 - s;
  let b1 = u2 * s;
  return vec3<f32>(b0, b1, 1.0 - b0 - b1);
}

fn ggx_visible_normal(v: vec3<f32>, alpha: f32, u1: f32, u2: f32) -> vec3<f32> {
  let vh = normalize(vec3<f32>(alpha * v.x, alpha * v.y, v.z));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  var t1 = vec3<f32>(1.0, 0.0, 0.0);
  if (lensq > 0.0) {
    t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq);
  }
  let t2 = cross(vh, t1);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let p1 = r * cos(phi);
  let s = 0.5 * (1.0 + vh.z);
  let p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * r * sin(phi);
  let nh = t1 * p1 + t2 * p2 + vh * sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2));
  let ne = vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z));
  return ne / max(length(ne), 1e-30);
}

fn power_heuristic(a: f32, b: f32) -> f32 {
  let a2 = a * a;
  let total = a2 + b * b;
  return select(0.0, a2 / total, total > 0.0);
}

fn max3(v: vec3<f32>) -> f32 {
  return max(v.x, max(v.y, v.z));
}

// -- surface points (pathtrace/scene.py) ------------------------------------------------------

struct Inter {
  inst: u32,
  prim: u32,
  mesh_tri: u32,     // the triangle in the concatenated mesh buffers
  material: u32,
  b: vec3<f32>,
  t: f32,            // distance from the ray's origin (unit rays)
  p: vec3<f32>,      // camera-relative world position
  ng: vec3<f32>,     // geometric normal, outward
  ns: vec3<f32>,     // interpolated shading normal, outward
  toward: f32,       // +1 / -1: turns outward normals toward the ray's origin
  d: vec3<f32>,
  dbdx: vec3<f32>,
  dbdy: vec3<f32>,
  width: f32,        // footprint width (the next ray cone starts with it)
  p0: vec3<f32>,
  p1: vec3<f32>,
  p2: vec3<f32>,
  uv0: vec2<f32>,
  uv1: vec2<f32>,
  uv2: vec2<f32>,
  has_uvs: bool,
  vn: vec3<f32>,     // interpolated normal, not renormalized (MikkTSpace N), outward
  tan: vec4<f32>,    // interpolated MikkTSpace tangent and sign (zero: none)
  color: vec4<f32>,  // interpolated vertex colour (white without vertex colours)
};

fn bary_derivative(e1: vec3<f32>, e2: vec3<f32>, dp: vec3<f32>) -> vec3<f32> {
  let a = dot(e1, e1);
  let b = dot(e1, e2);
  let c = dot(e2, e2);
  let det = a * c - b * b;
  if (abs(det) <= 1e-30) {
    return vec3<f32>(0.0);
  }
  let p1 = dot(e1, dp);
  let p2 = dot(e2, dp);
  let db1 = (c * p1 - b * p2) / det;
  let db2 = (a * p2 - b * p1) / det;
  return vec3<f32>(-db1 - db2, db1, db2);
}

struct Corners3 {
  p0: vec3<f32>,
  p1: vec3<f32>,
  p2: vec3<f32>,
};

// The triangle's corners and the shading data of a surface point.
fn triangle_corners(inst: u32, mesh_tri: u32) -> Corners3 {
  let tv = triangle_vertices(mesh_tri);
  let rec = instances[inst];
  return Corners3(
    relative_position(rec, mesh_position(tv.x)),
    relative_position(rec, mesh_position(tv.y)),
    relative_position(rec, mesh_position(tv.z))
  );
}

fn outward_normal(inst: u32, c: Corners3) -> vec3<f32> {
  let g = cross(c.p1 - c.p0, c.p2 - c.p0);
  let len = length(g);
  var ng = select(vec3<f32>(0.0), g / len, len > 0.0);
  if ((instances[inst].flags & INSTANCE_MIRRORED) != 0u) {
    ng = -ng;
  }
  return ng;
}

fn triangle_material_of(inst: u32, mesh_tri: u32) -> u32 {
  var group = 0u;
  if ((instances[inst].flags & INSTANCE_HAS_GROUPS) != 0u) {
    group = mesh_groups[mesh_tri];
  }
  return triangle_material(inst, group);
}

// A hit of the unit ray (o, d): `t` is the distance from o. Camera rays give differentials
// (do/dx, do/dy, dd/dx, dd/dy); other rays a cone (width at o, spread angle).
fn interaction(
  h: Hit, t: f32, d: vec3<f32>, has_diff: bool,
  do_dx: vec3<f32>, do_dy: vec3<f32>, dd_dx: vec3<f32>, dd_dy: vec3<f32>,
  cone_w: f32, cone_spread: f32
) -> Inter {
  var it: Inter;
  it.inst = h.inst;
  it.prim = h.prim;
  it.mesh_tri = pt[inst_base(h.inst) + 19u] + h.prim;
  it.b = h.b;
  it.t = t;
  it.d = d;
  let c = triangle_corners(h.inst, it.mesh_tri);
  it.p0 = c.p0;
  it.p1 = c.p1;
  it.p2 = c.p2;
  it.p = h.b.x * c.p0 + h.b.y * c.p1 + h.b.z * c.p2;
  let e1 = c.p1 - c.p0;
  let e2 = c.p2 - c.p0;
  it.ng = outward_normal(h.inst, c);
  let rec = instances[h.inst];
  let tv = triangle_vertices(it.mesh_tri);
  it.ns = it.ng;
  it.vn = it.ng;
  if ((rec.flags & INSTANCE_HAS_NORMALS) != 0u) {
    let n = h.b.x * safe_normalize(world_normal(rec, mesh_normal(tv.x)))
      + h.b.y * safe_normalize(world_normal(rec, mesh_normal(tv.y)))
      + h.b.z * safe_normalize(world_normal(rec, mesh_normal(tv.z)));
    let len = length(n);
    if (len > 1e-6) {
      it.ns = n / len;
      it.vn = n;
    }
  }
  it.toward = select(-1.0, 1.0, dot(it.ng, d) < 0.0);
  var dp_dx = vec3<f32>(0.0);
  var dp_dy = vec3<f32>(0.0);
  if (has_diff) {
    let denom = dot(d, it.ng);
    let safe = select(1.0, denom, abs(denom) > 1e-30);
    let vx = do_dx + t * dd_dx;
    dp_dx = vx + (-dot(vx, it.ng) / safe) * d;
    let vy = do_dy + t * dd_dy;
    dp_dy = vy + (-dot(vy, it.ng) / safe) * d;
    it.width = max(length(dp_dx), length(dp_dy));
  } else {
    it.width = cone_w + cone_spread * t;
    var ta = cross(d, select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(d.z) < 0.9));
    ta = ta / max(length(ta), 1e-30);
    let tb = cross(d, ta);
    let denom = dot(d, it.ng);
    let safe = select(select(1e-6, -1e-6, denom < 0.0), denom, abs(denom) > 1e-6);
    let vx = ta * it.width;
    dp_dx = vx - d * (dot(vx, it.ng) / safe);
    let vy = tb * it.width;
    dp_dy = vy - d * (dot(vy, it.ng) / safe);
  }
  it.dbdx = bary_derivative(e1, e2, dp_dx);
  it.dbdy = bary_derivative(e1, e2, dp_dy);
  it.has_uvs = (rec.flags & INSTANCE_HAS_UVS) != 0u;
  it.uv0 = vec2<f32>(0.0);
  it.uv1 = vec2<f32>(0.0);
  it.uv2 = vec2<f32>(0.0);
  if (it.has_uvs) {
    it.uv0 = mesh_uv(tv.x);
    it.uv1 = mesh_uv(tv.y);
    it.uv2 = mesh_uv(tv.z);
  }
  it.material = triangle_material_of(h.inst, it.mesh_tri);
  it.color = vertex_color(rec, tv, h.b);
  it.tan = vec4<f32>(0.0);
  if (it.has_uvs && needs_tangents(it.material)) {
    let c3 = 3u * it.mesh_tri;
    let t0 = world_tangent(rec, mesh_tangent(c3));
    let t1 = world_tangent(rec, mesh_tangent(c3 + 1u));
    let t2 = world_tangent(rec, mesh_tangent(c3 + 2u));
    it.tan = h.b.x * t0 + h.b.y * t1 + h.b.z * t2;
  }
  return it;
}

// The vertex colour at barycentrics `b` of a triangle with vertices `tv` (white without).
fn vertex_color(rec: Instance, tv: vec3<u32>, b: vec3<f32>) -> vec4<f32> {
  if ((rec.flags & INSTANCE_HAS_COLORS) == 0u) {
    return vec4<f32>(1.0);
  }
  return b.x * mesh_color(rec, tv.x) + b.y * mesh_color(rec, tv.y) + b.z * mesh_color(rec, tv.z);
}

fn surface_input(it: Inter) -> SurfaceInput {
  return SurfaceInput(it.inst, it.b, it.dbdx, it.dbdy, it.p0, it.p1, it.p2, it.uv0, it.uv1, it.uv2, it.ns, it.vn, it.tan, it.color);
}

// Ray origins off a surface along n (Waechter and Binder 2019): a fixed number of float
// ulps per coordinate, a small absolute step near the origin.
fn offset_origin(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let of_i = vec3<i32>(OFFSET_INT * n);
  let p_i = bitcast<vec3<f32>>(bitcast<vec3<i32>>(p) + select(of_i, -of_i, p < vec3<f32>(0.0)));
  return select(p_i, p + OFFSET_FLOAT * n, abs(p) < vec3<f32>(OFFSET_ORIGIN));
}

// -- the BSDF (pathtrace/bsdf.py) ----------------------------------------------------------------

struct Bsdf {
  f: Basis,
  n: vec3<f32>,
  wo: vec3<f32>,      // local, z > 0
  c_diff: vec3<f32>,
  f0: vec3<f32>,
  alpha: f32,
  specular: bool,
  p_spec: f32,
};

fn bsdf_build(n: vec3<f32>, wo_world: vec3<f32>, base: vec3<f32>, metallic_in: f32, roughness: f32, specular: f32, has_specular: bool) -> Bsdf {
  let metallic = clamp(metallic_in, 0.0, 1.0);
  let rough = clamp(roughness, MIN_ROUGHNESS, 1.0);
  let c_diff = base * (1.0 - metallic);
  var f0 = vec3<f32>(0.08 * specular * (1.0 - metallic)) + base * metallic;
  if (!has_specular) {
    f0 = vec3<f32>(0.0);
  }
  let f = basis(n);
  var wo = to_local(wo_world, f, n);
  wo.z = max(wo.z, MIN_COS);
  let f_view = f0 + (1.0 - f0) * pow(max(1.0 - wo.z, 0.0), 5.0);
  let w_spec = select(0.0, luminance(f_view), has_specular);
  let w_diff = luminance(c_diff) * (1.0 - w_spec);
  let total = w_spec + w_diff;
  var p = select(0.0, w_spec / total, total > 0.0);
  if (has_specular && w_diff > 0.0) {
    p = clamp(p, LOBE_MIN, LOBE_MAX);
  } else {
    p = select(0.0, 1.0, has_specular);
  }
  return Bsdf(f, n, wo, c_diff, f0, rough * rough, has_specular, p);
}

struct BsdfValue {
  f: vec3<f32>,
  pdf: f32,
};

fn bsdf_eval_local(s: Bsdf, wi: vec3<f32>) -> BsdfValue {
  let wo = s.wo;
  let ndl = wi.z;
  let ndv = wo.z;
  if (!(ndl > MIN_COS)) {
    return BsdfValue(vec3<f32>(0.0), 0.0);
  }
  let a2 = s.alpha * s.alpha;
  let h = (wi + wo) / max(length(wi + wo), 1e-30);
  let ndh = max(h.z, 0.0);
  let vdh = clamp(dot(wo, h), 0.0, 1.0);
  var fresnel = s.f0 + (1.0 - s.f0) * pow(1.0 - vdh, 5.0);
  if (!s.specular) {
    fresnel = vec3<f32>(0.0);
  }
  let dd = ndh * ndh * (a2 - 1.0) + 1.0;
  let dist = a2 / (PI * dd * dd);
  let vis = 0.5 / max(ndl * sqrt(ndv * ndv * (1.0 - a2) + a2) + ndv * sqrt(ndl * ndl * (1.0 - a2) + a2), 1e-30);
  let f = (1.0 - fresnel) * s.c_diff / PI + fresnel * (dist * vis);
  let g1 = 2.0 * ndv / (ndv + sqrt(a2 + (1.0 - a2) * ndv * ndv));
  let pdf_spec = g1 * dist / (4.0 * ndv);
  let pdf = s.p_spec * pdf_spec + (1.0 - s.p_spec) * ndl / PI;
  return BsdfValue(f, pdf);
}

struct BsdfEval {
  f: vec3<f32>,
  pdf: f32,
  cos: f32,
};

fn bsdf_eval(s: Bsdf, wi_world: vec3<f32>) -> BsdfEval {
  let wi = to_local(wi_world, s.f, s.n);
  let v = bsdf_eval_local(s, wi);
  return BsdfEval(v.f, v.pdf, max(wi.z, 0.0));
}

struct BsdfSample {
  wi: vec3<f32>,     // world
  weight: vec3<f32>, // f cos / pdf
  pdf: f32,
  valid: bool,
  specular: bool,
};

fn bsdf_sample(s: Bsdf, u_lobe: f32, u1: f32, u2: f32) -> BsdfSample {
  let spec = u_lobe < s.p_spec;
  var wi = cosine_hemisphere(u1, u2);
  if (spec) {
    let h = ggx_visible_normal(s.wo, s.alpha, u1, u2);
    wi = 2.0 * dot(s.wo, h) * h - s.wo;
  }
  let v = bsdf_eval_local(s, wi);
  let valid = wi.z > MIN_COS && v.pdf > 0.0;
  var weight = vec3<f32>(0.0);
  if (valid) {
    weight = v.f * (wi.z / v.pdf);
  }
  return BsdfSample(to_world(wi, s.f, s.n), weight, v.pdf, valid, spec);
}

fn bsdf_albedo(s: Bsdf) -> vec3<f32> {
  let f_view = s.f0 + (1.0 - s.f0) * pow(max(1.0 - s.wo.z, 0.0), 5.0);
  return s.c_diff * (1.0 - f_view) + f_view;
}

// -- the environment (pathtrace/environment.py) -----------------------------------------------

fn env_radiance(d: vec3<f32>) -> vec3<f32> {
  return env_base(d) + atmo_glow(params.atmo, d);
}

fn env_base(d: vec3<f32>) -> vec3<f32> {
  if (params.env_kind == ENV_IMAGE) {
    return env_image_radiance(d);
  }
  if (params.env_kind != ENV_SKY) {
    return params.env_uniform;
  }
  let z = clamp(d.z, -1.0, 1.0);
  if (z >= 0.0) {
    return params.horizon + (params.zenith - params.horizon) * sqrt(max(z, 0.0));
  }
  return params.horizon + (params.ground - params.horizon) * min(1.0, 4.0 * max(-z, 0.0));
}

fn env_flags() -> u32 {
  return bitcast<u32>(params.env_image.w);
}

fn env_sampled() -> bool {
  return params.env_kind == ENV_IMAGE && (env_flags() & ENV_SAMPLED) != 0u;
}

fn env_dims() -> vec4<u32> {  // width, height, cells across (nu), cells down (nv)
  let b = pt[H_ENV];
  return vec4<u32>(pt[b], pt[b + 1u], pt[b + 2u], pt[b + 3u]);
}

fn env_pixel(x: u32, y: u32, w: u32) -> vec3<f32> {
  return ptv3(pt[H_ENV] + 4u + 3u * (y * w + x));
}

// Image environments (envimage.py): R_z(rotation) d, Blender's equirectangular mapping,
// bilinear, wrapping across u.
fn env_uv(d: vec3<f32>) -> vec2<f32> {
  let c = params.env_image.x;
  let s = params.env_image.y;
  let e = vec3<f32>(c * d.x - s * d.y, s * d.x + c * d.y, d.z);
  var u = 0.5 - atan2(e.y, e.x) / (2.0 * PI);
  u = u - floor(u);
  let v = 0.5 + asin(clamp(e.z, -1.0, 1.0)) / PI;
  return vec2<f32>(u, v);
}

fn env_image_radiance(d: vec3<f32>) -> vec3<f32> {
  let dims = env_dims();
  let w = i32(dims.x);
  let h = i32(dims.y);
  let uv = env_uv(d);
  let x = uv.x * f32(w) - 0.5;
  let y = (1.0 - uv.y) * f32(h) - 0.5;
  let x0f = floor(x);
  let y0f = floor(y);
  let fx = x - x0f;
  let fy = y - y0f;
  let x0 = u32(((i32(x0f) % w) + w) % w);
  let x1 = (x0 + 1u) % u32(w);
  let y0 = u32(clamp(i32(y0f), 0, h - 1));
  let y1 = u32(clamp(i32(y0f) + 1, 0, h - 1));
  let top = env_pixel(x0, y0, u32(w)) * (1.0 - fx) + env_pixel(x1, y0, u32(w)) * fx;
  let bottom = env_pixel(x0, y1, u32(w)) * (1.0 - fx) + env_pixel(x1, y1, u32(w)) * fx;
  return (top * (1.0 - fy) + bottom * fy) * params.env_image.z;
}

struct EnvSample {
  d: vec3<f32>,
  pdf: f32,
};

// A direction drawn by the image's luminance (EnvironmentImage.sample): u0 picks the row,
// u1 the column, their remainders the point in the cell.
fn env_sample(u0: f32, u1: f32) -> EnvSample {
  let dims = env_dims();
  let nu = dims.z;
  let nv = dims.w;
  let mb = pt[H_ENV] + 4u + 3u * dims.x * dims.y;  // marginal CDF (nv + 1)
  let cb = mb + nv + 1u;                            // conditional CDFs (nv x (nu + 1))
  var lo = 0u;
  var hi = nv - 1u;
  while (lo < hi) {  // the last row whose CDF starts at or below u0
    let mid = (lo + hi + 1u) / 2u;
    if (ptf(mb + mid) <= u0) {
      lo = mid;
    } else {
      hi = mid - 1u;
    }
  }
  let r = lo;
  let pr = ptf(mb + r + 1u) - ptf(mb + r);
  let fr = clamp((u0 - ptf(mb + r)) / select(1.0, pr, pr > 0.0), 0.0, 1.0);
  let rb = cb + r * (nu + 1u);
  lo = 0u;
  hi = nu - 1u;
  while (lo < hi) {
    let mid = (lo + hi + 1u) / 2u;
    if (ptf(rb + mid) <= u1) {
      lo = mid;
    } else {
      hi = mid - 1u;
    }
  }
  let c = lo;
  let pc = ptf(rb + c + 1u) - ptf(rb + c);
  let fc = clamp((u1 - ptf(rb + c)) / select(1.0, pc, pc > 0.0), 0.0, 1.0);
  let u = (f32(c) + fc) / f32(nu);
  let v = 1.0 - (f32(r) + fr) / f32(nv);
  let el = (v - 0.5) * PI;
  let az = (0.5 - u) * 2.0 * PI;
  let ce = cos(el);
  let e = vec3<f32>(ce * cos(az), ce * sin(az), sin(el));
  let cr = params.env_image.x;
  let sr = params.env_image.y;
  let d = vec3<f32>(cr * e.x + sr * e.y, -sr * e.x + cr * e.y, e.z);
  var pdf = 0.0;
  if (ce > 1e-9) {
    pdf = pr * pc * f32(nu) * f32(nv) / (2.0 * PI * PI * ce);
  }
  return EnvSample(d, pdf);
}

// The density env_sample draws direction d with (EnvironmentImage.pdf).
fn env_pdf(d: vec3<f32>) -> f32 {
  let dims = env_dims();
  let nu = dims.z;
  let nv = dims.w;
  let mb = pt[H_ENV] + 4u + 3u * dims.x * dims.y;
  let cb = mb + nv + 1u;
  let uv = env_uv(d);
  let r = u32(clamp(i32((1.0 - uv.y) * f32(nv)), 0, i32(nv) - 1));
  let c = u32(clamp(i32(uv.x * f32(nu)), 0, i32(nu) - 1));
  let rb = cb + r * (nu + 1u);
  let p = (ptf(mb + r + 1u) - ptf(mb + r)) * (ptf(rb + c + 1u) - ptf(rb + c)) * f32(nu) * f32(nv);
  let ce = cos((uv.y - 0.5) * PI);
  if (ce <= 1e-9) {
    return 0.0;
  }
  return p / (2.0 * PI * PI * ce);
}

fn env_camera(d: vec3<f32>) -> vec3<f32> {
  if (params.env_kind == ENV_AMBIENT || (params.env_kind == ENV_IMAGE && (env_flags() & ENV_VISIBLE) == 0u)) {
    return params.background + atmo_glow(params.atmo, d);
  }
  return env_radiance(d);
}

// -- lights (pathtrace/lights.py) ------------------------------------------------------------------
// Light record: 0 kind (0 directional, 1 point, 2 spot, 3 area), 1 sized, 2 inverse square,
// 3 has range, 4-6 radiance (c I), 7 range, 8-10 position (camera-relative), 11 radius,
// 12-14 toward the light, 15 cos(half angle), 16 1 - cos, 17 cone pdf, 18-20 disk radiance,
// 21 disk scale, 22 casts no shadow, 24-26 spot axis / area normal (the way they shine),
// 27 spot cos_outer, 28 spot smoothstep width, 29 area shape (1: disk or ellipse), 30 area,
// 31-33 area half extent x, 34-36 half extent y, 37-39 area radiance (pi c I / A),
// 40 tan(spread / 2) (0: 180 degrees), 41 the spread's normalization.

fn light_base(l: u32) -> u32 {
  return pt[H_LIGHTS] + l * LIGHT_WORDS;
}

fn point_scale(lb: u32, dist: f32) -> vec3<f32> {
  let d = max(dist, MIN_LIGHT_DISTANCE);
  var atten = 1.0;
  if (pt[lb + 2u] != 0u) {
    atten = 1.0 / (d * d);
  }
  if (pt[lb + 3u] != 0u) {
    let ratio = d / ptf(lb + 7u);
    let w = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    atten = atten * w * w;
  }
  return PI * ptv3(lb + 4u) * atten;
}

// Light linking: whether light l leaves instance inst unlit (instance words 21, 22).
fn unlinked(inst: u32, l: u32) -> bool {
  if (l >= 64u) {
    return false;
  }
  let word = pt[inst_base(inst) + 21u + l / 32u];
  return ((word >> (l % 32u)) & 1u) != 0u;
}

// A spot's cone mask at the cosine between its axis and the direction from it.
fn spot_mask(lb: u32, cos_t: f32) -> f32 {
  let cos_outer = ptf(lb + 27u);
  if (cos_t <= cos_outer) {
    return 0.0;
  }
  let smooth_w = ptf(lb + 28u);
  if (smooth_w > 0.0) {
    let x = clamp((cos_t - cos_outer) / smooth_w, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
  }
  return 1.0;
}

// A spread area light's radiance factor toward cos_a from its normal (scene_data.py).
fn spread_factor(lb: u32, cos_a: f32) -> f32 {
  if (!(cos_a > 0.0)) {
    return 0.0;
  }
  let tan_h = ptf(lb + 40u);
  if (tan_h <= 0.0) {
    return 1.0;
  }
  let tan_a = sqrt(max(1.0 - cos_a * cos_a, 0.0)) / max(cos_a, 1e-30);
  return max(1.0 - tan_a / tan_h, 0.0) * ptf(lb + 41u);
}

// The part of spread light lb that reaches p (RenderLight.spread_window): (u0, w0, u1, w1)
// in half extents, and its area (0: none).
struct Window {
  lohi: vec4<f32>,
  area: f32,
};

fn spread_window(lb: u32, p: vec3<f32>) -> Window {
  let rel = p - ptv3(lb + 8u);
  let t = dot(rel, ptv3(lb + 24u));
  let ax = ptv3(lb + 31u);
  let ay = ptv3(lb + 34u);
  let lx = length(ax);
  let ly = length(ay);
  let qu = dot(rel, ax) / (lx * lx);
  let qw = dot(rel, ay) / (ly * ly);
  let reach = max(t, 0.0) * ptf(lb + 40u);
  let lohi = vec4<f32>(
    max(qu - reach / lx, -1.0), max(qw - reach / ly, -1.0),
    min(qu + reach / lx, 1.0), min(qw + reach / ly, 1.0)
  );
  var area = 0.0;
  if (t > 0.0 && lohi.z > lohi.x && lohi.w > lohi.y) {
    area = (lohi.z - lohi.x) * lx * (lohi.w - lohi.y) * ly;
  }
  return Window(lohi, area);
}

struct LightSample {
  wi: vec3<f32>,
  goal: vec3<f32>,
  has_target: bool,  // false: the light is infinitely far
  scale: vec3<f32>,  // unoccluded irradiance per unit cos
  pdf: f32,
  delta: bool,       // no size: light sampling is the only strategy
};

fn sample_analytic(l: u32, p: vec3<f32>, key: vec2<u32>, vertex: u32, index: u32) -> LightSample {
  let lb = light_base(l);
  let slot = SLOT_LIGHTS + 3u * index;
  var out: LightSample;
  if (pt[lb] == 3u && ptf(lb + 40u) > 0.0) {  // a spread light: the part reaching p
    let u1 = rnd(key, vertex, slot);
    let u2 = rnd(key, vertex, slot + 1u);
    let win = spread_window(lb, p);
    let a = win.lohi.x + (win.lohi.z - win.lohi.x) * u1;
    let b = win.lohi.y + (win.lohi.w - win.lohi.y) * u2;
    var reach = win.area > 0.0;
    if (pt[lb + 29u] != 0u) {
      reach = reach && a * a + b * b <= 1.0;
    }
    let x = ptv3(lb + 8u) + a * ptv3(lb + 31u) + b * ptv3(lb + 34u);
    let v = x - p;
    let d2 = dot(v, v);
    let dist = sqrt(d2);
    let wi = v / max(dist, 1e-30);
    let cos_l = -dot(wi, ptv3(lb + 24u));
    let g = spread_factor(lb, cos_l);
    let ok = reach && cos_l > 1e-12 && dist > 0.0 && g > 0.0;
    out.wi = wi;
    out.goal = x;
    out.has_target = true;
    out.delta = !ok;
    out.pdf = select(0.0, d2 / (win.area * cos_l), ok);
    out.scale = select(vec3<f32>(0.0), ptv3(lb + 37u) * (g * win.area * cos_l / d2), ok);
    return out;
  }
  if (pt[lb] == 3u) {  // an area light: a uniform point of its shape
    let u1 = rnd(key, vertex, slot);
    let u2 = rnd(key, vertex, slot + 1u);
    var ab = vec2<f32>(2.0 * u1 - 1.0, 2.0 * u2 - 1.0);
    if (pt[lb + 29u] != 0u) {
      ab = concentric_disk(u1, u2);
    }
    let x = ptv3(lb + 8u) + ab.x * ptv3(lb + 31u) + ab.y * ptv3(lb + 34u);
    let v = x - p;
    let d2 = dot(v, v);
    let dist = sqrt(d2);
    let wi = v / max(dist, 1e-30);
    let cos_l = -dot(wi, ptv3(lb + 24u));
    let ok = cos_l > 1e-12 && dist > 0.0;
    out.wi = wi;
    out.goal = x;
    out.has_target = true;
    out.delta = !ok;  // (not sampleable from here: weight 1, contributes 0)
    out.pdf = select(0.0, d2 / (ptf(lb + 30u) * cos_l), ok);
    out.scale = select(vec3<f32>(0.0), PI * ptv3(lb + 4u) * (cos_l / d2), ok);
    return out;
  }
  if (pt[lb] == 0u) {
    let axis = ptv3(lb + 12u);
    out.has_target = false;
    out.goal = vec3<f32>(0.0);
    if (pt[lb + 1u] != 0u) {
      let local = uniform_cone(rnd(key, vertex, slot), rnd(key, vertex, slot + 1u), ptf(lb + 16u));
      out.wi = to_world(local, basis(axis), axis);
      out.pdf = ptf(lb + 17u);
      out.delta = false;
    } else {
      out.wi = axis;
      out.pdf = 0.0;
      out.delta = true;
    }
    out.scale = PI * ptv3(lb + 4u) * ptf(lb + 21u);
    return out;
  }
  let center = ptv3(lb + 8u);
  let v = center - p;
  let dist = length(v);
  let axis = v / max(dist, 1e-30);
  let r = ptf(lb + 11u);
  out.goal = center;
  out.has_target = true;
  out.wi = axis;
  out.pdf = 0.0;
  out.delta = true;
  var sphere = 1.0;
  if (r > 0.0 && dist > r) {
    let sin2 = (r / dist) * (r / dist);
    let cmax = sqrt(max(0.0, 1.0 - sin2));
    let omc = sin2 / (1.0 + cmax);
    let local = uniform_cone(rnd(key, vertex, slot), rnd(key, vertex, slot + 1u), omc);
    let w = to_world(local, basis(axis), axis);
    let proj = dot(w, v);
    let disc = max(proj * proj - (dist * dist - r * r), 0.0);
    let t_hit = max(proj - sqrt(disc), 0.0);
    out.wi = w;
    out.goal = p + w * t_hit;
    sphere = 2.0 * omc / sin2;
    out.pdf = 1.0 / (2.0 * PI * max(omc, 1e-30));
    out.delta = false;
  }
  out.scale = point_scale(lb, dist) * sphere;
  if (pt[lb] == 2u) {
    out.scale = out.scale * spot_mask(lb, -dot(axis, ptv3(lb + 24u)));
  }
  return out;
}

// Where a ray (o, unit d) enters area light lb's front face inside its shape (and, for
// spread lights, inside the part of it that reaches o) before t_limit: the light's
// solid-angle density there and its radiance factor, or 0s.
fn area_hit(lb: u32, o: vec3<f32>, d: vec3<f32>, t_limit: f32) -> vec2<f32> {
  let n = ptv3(lb + 24u);
  let denom = dot(d, n);
  if (!(denom < 0.0)) {
    return vec2<f32>(0.0);
  }
  let c = ptv3(lb + 8u);
  let t = dot(c - o, n) / denom;
  if (!(t > 0.0 && t < t_limit)) {
    return vec2<f32>(0.0);
  }
  let q = o + d * t - c;
  let ax = ptv3(lb + 31u);
  let ay = ptv3(lb + 34u);
  let u = dot(q, ax) / dot(ax, ax);
  let w = dot(q, ay) / dot(ay, ay);
  var inside = abs(u) <= 1.0 && abs(w) <= 1.0;
  if (pt[lb + 29u] != 0u) {
    inside = u * u + w * w <= 1.0;
  }
  if (!inside) {
    return vec2<f32>(0.0);
  }
  let cos_l = max(-denom, 1e-30);
  if (ptf(lb + 40u) > 0.0) {
    let win = spread_window(lb, o);
    let in_win = win.area > 0.0 && u >= win.lohi.x && u <= win.lohi.z && w >= win.lohi.y && w <= win.lohi.w;
    let g = spread_factor(lb, cos_l);
    if (!in_win || !(g > 0.0)) {
      return vec2<f32>(0.0);
    }
    return vec2<f32>(t * t / (win.area * cos_l), g);
  }
  return vec2<f32>(t * t / (ptf(lb + 30u) * cos_l), 1.0);
}

// Radiance of sized lights a BSDF ray (origin o, unit d) reaches before geometry at
// t_geometry (PT_INF: none), weighted against light sampling.
fn hit_sized(o: vec3<f32>, d: vec3<f32>, t_geometry: f32, pdf_bsdf: f32, from_inst: u32) -> vec3<f32> {
  var out = vec3<f32>(0.0);
  for (var l = 0u; l < params.light_count; l++) {
    let lb = light_base(l);
    if (pt[lb + 1u] == 0u || unlinked(from_inst, l)) {
      continue;
    }
    // Lights without shadows are reached through surfaces (lights.py).
    var limit = t_geometry;
    if (pt[lb + 22u] != 0u) {
      limit = PT_INF;
    }
    let escaped = limit >= PT_INF;
    if (pt[lb] == 3u) {
      let hit = area_hit(lb, o, d, limit);
      if (hit.x > 0.0) {
        out += ptv3(lb + 37u) * (power_heuristic(pdf_bsdf, hit.x) * hit.y);
      }
      continue;
    }
    if (pt[lb] == 0u) {
      if (escaped && dot(d, ptv3(lb + 12u)) >= ptf(lb + 15u)) {
        out += ptv3(lb + 18u) * power_heuristic(pdf_bsdf, ptf(lb + 17u));
      }
      continue;
    }
    let r = ptf(lb + 11u);
    let v = ptv3(lb + 8u) - o;
    let dist2 = dot(v, v);
    if (!(dist2 > r * r)) {
      continue;
    }
    let proj = dot(d, v);
    let disc = proj * proj - (dist2 - r * r);
    if (disc < 0.0) {
      continue;
    }
    let t_sphere = proj - sqrt(max(disc, 0.0));
    if (t_sphere > 0.0 && t_sphere < limit) {
      let sin2 = r * r / dist2;
      let cmax = sqrt(max(0.0, 1.0 - sin2));
      let pdf = 1.0 / (2.0 * PI * max(sin2 / (1.0 + cmax), 1e-30));
      let dist = sqrt(dist2);
      var radiance = point_scale(lb, dist) / (PI * sin2);
      if (pt[lb] == 2u) {
        radiance = radiance * spot_mask(lb, dot(-v / dist, ptv3(lb + 24u)));
      }
      out += radiance * power_heuristic(pdf_bsdf, pdf);
    }
  }
  return out;
}

// Emitter record: 0 instance, 1 mesh triangle of the instance's mesh, 2 material,
// 3 double-sided, 4-6 outward normal, 7 area, 8 selection probability.

fn emitter_base(k: u32) -> u32 {
  return pt[H_EMITTERS] + k * EMITTER_WORDS;
}

// The emitter picked by u: the first whose cumulative probability exceeds u.
fn pick_emitter(u: f32) -> u32 {
  let base = pt[H_CDF];
  var lo = 0u;
  var hi = params.emitter_count;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    if (ptf(base + mid) <= u) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  return min(lo, params.emitter_count - 1u);
}

// Solid-angle density with which light sampling picks a hit point of an emitter.
fn emitter_pdf(it: Inter) -> f32 {
  let gtri = pt[inst_base(it.inst) + 20u] + it.prim;
  let k = pt[pt[H_LOOKUP] + gtri];
  if (k == PT_NONE) {
    return 0.0;
  }
  let eb = emitter_base(k);
  let cos_l = abs(dot(it.ng, it.d));
  if (!(cos_l > 1e-12)) {
    return 0.0;
  }
  return ptf(eb + 8u) / ptf(eb + 7u) * it.t * it.t / cos_l;
}

// -- one path: a state machine with one material evaluation ---------------------------------

const PH_TRACE: u32 = 0u;
const PH_HIT: u32 = 1u;
const PH_ALPHA: u32 = 2u;
const PH_SHADE: u32 = 3u;
const PH_NEE: u32 = 4u;
const PH_EMIT: u32 = 5u;
const PH_BOUNCE: u32 = 6u;
const PH_AO: u32 = 7u;
const PH_DONE: u32 = 8u;

const RAY_CAMERA: u32 = 0u;
const RAY_BOUNCE: u32 = 1u;
const RAY_SHADOW: u32 = 2u;
const RAY_AO: u32 = 3u;
const RAY_CENTER: u32 = 4u;

struct State {
  key: vec2<u32>,
  phase: u32,
  // The ray in flight: its current segment, and where it started (pass-through layers).
  kind: u32,
  ro: vec3<f32>,
  rd: vec3<f32>,
  cam_d: vec3<f32>,  // the camera ray's direction (fog)
  first_t: f32,      // distance to the first surface the camera ray reached (-1: none)
  rtmin: f32,
  rtmax: f32,
  skip_inst: u32,
  skip_prim: u32,
  origin: vec3<f32>,
  travel: f32,
  layers: u32,
  has_diff: bool,
  do_dx: vec3<f32>,
  do_dy: vec3<f32>,
  dd_dx: vec3<f32>,
  dd_dy: vec3<f32>,
  cone_w: f32,
  cone_spread: f32,
  hit: Hit,
  cand: Inter,        // the surface the ray just hit
  // Material evaluation requested for the next loop iteration.
  eval_pending: bool,
  eval_alpha: bool,
  eval_m: u32,
  eval_in: SurfaceInput,
  eval_uvs: bool,
  sv: SurfaceValues,
  // The path.
  vertex: u32,
  throughput: vec3<f32>,
  prev_pdf: f32,
  spread: f32,
  it: Inter,          // the vertex being shaded
  bsdf: Bsdf,
  ng_side: vec3<f32>,
  leave: vec3<f32>,   // ray origin off the vertex
  // Next-event estimation at the vertex.
  nee_light: u32,
  nee_k: u32,
  nee_index: u32,
  nee_sum: vec3<f32>,
  pending: vec3<f32>,
  em_wi: vec3<f32>,
  em_x: vec3<f32>,
  em_pdf: f32,
  em_ok: bool,
  em_masked: bool,
  em_cutoff: f32,
  // Ambient occlusion at the camera hit.
  ao_j: u32,
  ao_blocked: f32,
  // Results.
  direct: vec3<f32>,
  indirect: vec3<f32>,
  albedo: vec3<f32>,
  normal: vec3<f32>,
  ao: f32,
  hit_any: bool,
  depth: f32,
  center_inst: u32,
  tally: Tally,
};

fn add_target(st: ptr<function, State>, v: vec3<f32>, direct: bool) {
  if (direct) {
    (*st).direct += v;
  } else {
    (*st).indirect += v;
  }
}

fn request_eval(st: ptr<function, State>, m: u32, s: SurfaceInput, has_uvs: bool, alpha_only: bool) {
  (*st).eval_pending = true;
  (*st).eval_alpha = alpha_only;
  (*st).eval_m = m;
  (*st).eval_in = s;
  (*st).eval_uvs = has_uvs;
}

fn start_ray(st: ptr<function, State>, kind: u32, o: vec3<f32>, d: vec3<f32>, tmin: f32, tmax: f32, skip_inst: u32, skip_prim: u32) {
  (*st).kind = kind;
  (*st).ro = o;
  (*st).rd = d;
  (*st).rtmin = tmin;
  (*st).rtmax = tmax;
  (*st).skip_inst = skip_inst;
  (*st).skip_prim = skip_prim;
  (*st).origin = o;
  (*st).travel = 0.0;
  (*st).layers = 0u;
  (*st).has_diff = false;
  (*st).cone_w = 0.0;
  (*st).cone_spread = 0.0;
  (*st).phase = PH_TRACE;
}

// A point of the unit aperture (camera.py's aperture_points, in float32): round (Shirley and
// Chiu's concentric mapping) or a polygon of `blades` corners inscribed in it, a corner up,
// turned and narrowed by the camera's rotation and ratio.
// Shirley and Chiu's concentric map of the unit square onto the unit disk.
fn concentric_disk(u1: f32, u2: f32) -> vec2<f32> {
  let sx = 2.0 * u1 - 1.0;
  let sy = 2.0 * u2 - 1.0;
  if (sx == 0.0 && sy == 0.0) {
    return vec2<f32>(0.0);
  }
  if (abs(sx) > abs(sy)) {
    let phi = (PI / 4.0) * (sy / sx);
    return vec2<f32>(sx * cos(phi), sx * sin(phi));
  }
  let phi = PI / 2.0 - (PI / 4.0) * (sx / sy);
  return vec2<f32>(sy * cos(phi), sy * sin(phi));
}

fn aperture_point(u1: f32, u2: f32) -> vec2<f32> {
  var p: vec2<f32>;
  let blades = params.aperture.x;
  if (blades >= 3.0) {
    let k = min(floor(u1 * blades), blades - 1.0);
    let a = u1 * blades - k;
    let step = 2.0 * PI / blades;
    let a0 = PI * 0.5 + k * step;
    let r = sqrt(a);
    let b1 = u2 * r;
    let b0 = r - b1;
    p = vec2<f32>(b0 * cos(a0) + b1 * cos(a0 + step), b0 * sin(a0) + b1 * sin(a0 + step));
  } else {
    p = concentric_disk(u1, u2);
  }
  let c = params.aperture.y;
  let s = params.aperture.z;
  p = vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
  return vec2<f32>(p.x * params.aperture.w, p.y);
}

fn camera_ray(st: ptr<function, State>, kind: u32, x: u32, y: u32, u: f32, v: f32, with_diff: bool, lu: f32, lv: f32) {
  let w = f32(params.width);
  let h = f32(params.height);
  let xn = (f32(x) + u) * (2.0 / w) - 1.0 + params.lens.x;
  let yn = 1.0 - (f32(y) + v) * (2.0 / h) + params.lens.y;
  let sx = xn * params.hx;
  let sy = yn * params.hy;
  let step_x = (2.0 / w) * params.hx * params.diff_scale;
  let step_y = -(2.0 / h) * params.hy * params.diff_scale;
  if ((params.flags & FLAG_PERSPECTIVE) != 0u) {
    let raw = params.forward + sx * params.right + sy * params.up;
    let len = length(raw);
    let unit = raw / len;
    if (params.lens.z > 0.0 && kind == RAY_CAMERA) {
      // A thin lens: from a point of the aperture toward where the pinhole ray meets the
      // plane in focus (raw has a unit forward component); differentials stay the pinhole's.
      let a = aperture_point(lu, lv) * params.lens.z;
      let offset = a.x * params.right + a.y * params.up;
      let lens_d = raw * params.lens.w - offset;
      let lens_unit = lens_d / length(lens_d);
      let along = dot(lens_unit, params.forward);
      start_ray(st, kind, offset, lens_unit, params.near / along, params.far / along, PT_NONE, PT_NONE);
    } else {
      start_ray(st, kind, vec3<f32>(0.0), unit, params.near * len, params.far * len, PT_NONE, PT_NONE);
    }
    let vx = params.right * step_x;
    let vy = params.up * step_y;
    (*st).do_dx = vec3<f32>(0.0);
    (*st).do_dy = vec3<f32>(0.0);
    (*st).dd_dx = (vx - unit * dot(unit, vx)) / len;
    (*st).dd_dy = (vy - unit * dot(unit, vy)) / len;
  } else {
    start_ray(st, kind, sx * params.right + sy * params.up, params.forward, params.near, params.far, PT_NONE, PT_NONE);
    (*st).do_dx = params.right * step_x;
    (*st).do_dy = params.up * step_y;
    (*st).dd_dx = vec3<f32>(0.0);
    (*st).dd_dy = vec3<f32>(0.0);
  }
  (*st).has_diff = with_diff;
}

fn scene_has_pass_through() -> bool {
  return (params.flags & FLAG_PASS_THROUGH) != 0u;
}

// The ray left the scene (or passed through too many surfaces).
fn on_miss(st: ptr<function, State>) {
  switch ((*st).kind) {
    case 0u: {  // RAY_CAMERA
      (*st).direct = env_camera((*st).rd);
      (*st).phase = PH_DONE;
    }
    case 1u: {  // RAY_BOUNCE: the environment, and suns the ray found
      let direct = (*st).vertex == 1u;
      var le = env_radiance((*st).rd);
      if (env_sampled()) {  // also sampled as a light: weight both strategies
        le = le * power_heuristic((*st).prev_pdf, env_pdf((*st).rd));
      }
      add_target(st, (*st).throughput * le, direct);
      if ((params.flags & FLAG_SIZED_LIGHTS) != 0u) {
        add_target(st, (*st).throughput * hit_sized((*st).origin, (*st).rd, PT_INF, (*st).prev_pdf, (*st).it.inst), direct);
      }
      (*st).phase = PH_DONE;
    }
    case 2u: {  // RAY_SHADOW: the light is visible
      (*st).nee_sum += (*st).pending;
      (*st).phase = PH_NEE;
    }
    case 3u: {  // RAY_AO: open
      (*st).phase = PH_AO;
    }
    default: {  // RAY_CENTER
      (*st).phase = PH_DONE;
    }
  }
}

// Continue the ray behind a transparent hit.
fn continue_ray(st: ptr<function, State>) {
  let step = (*st).hit.t;
  (*st).ro = (*st).ro + (*st).rd * step;
  (*st).travel += step;
  (*st).rtmin = 0.0;
  if ((*st).rtmax < PT_INF) {
    (*st).rtmax = (*st).rtmax - step;
  }
  (*st).skip_inst = (*st).hit.inst;
  (*st).skip_prim = (*st).hit.prim;
  (*st).layers += 1u;
  (*st).tally.through += 1u;
  if ((*st).layers >= MAX_LAYERS) {
    (*st).tally.limited += 1u;
    on_miss(st);
    return;
  }
  (*st).phase = PH_TRACE;
}

// The ray stops at the surface it hit.
fn on_accept(st: ptr<function, State>) {
  switch ((*st).kind) {
    case 0u: {  // RAY_CAMERA
      (*st).hit_any = true;
      atomicAdd(&counters[C_VISIBLE + (*st).cand.inst], 1u);
      (*st).it = (*st).cand;
      if ((params.flags & FLAG_AO) != 0u) {
        (*st).ao_j = 0u;
        (*st).ao_blocked = 0.0;
        (*st).phase = PH_AO;
      } else {
        request_eval(st, (*st).it.material, surface_input((*st).it), (*st).it.has_uvs, false);
        (*st).phase = PH_SHADE;
      }
    }
    case 1u: {  // RAY_BOUNCE: lights in front of the surface, then shade it
      if ((params.flags & FLAG_SIZED_LIGHTS) != 0u) {
        add_target(
          st,
          (*st).throughput * hit_sized((*st).origin, (*st).rd, (*st).cand.t, (*st).prev_pdf, (*st).it.inst),
          (*st).vertex == 1u
        );
      }
      (*st).it = (*st).cand;
      request_eval(st, (*st).it.material, surface_input((*st).it), (*st).it.has_uvs, false);
      (*st).phase = PH_SHADE;
    }
    case 2u: {  // RAY_SHADOW: occluded
      (*st).phase = PH_NEE;
    }
    case 3u: {  // RAY_AO: blocked
      (*st).ao_blocked += 1.0;
      (*st).phase = PH_AO;
    }
    default: {  // RAY_CENTER
      (*st).depth = (*st).cand.t * dot((*st).rd, params.forward);
      (*st).center_inst = (*st).cand.inst;
      (*st).phase = PH_DONE;
    }
  }
}

fn on_hit(st: ptr<function, State>) {
  let h = (*st).hit;
  if (h.inst == PT_NONE) {
    on_miss(st);
    return;
  }
  let camera = (*st).kind == RAY_CAMERA || (*st).kind == RAY_CENTER;
  (*st).cand = interaction(
    h, (*st).travel + h.t, (*st).rd, (*st).has_diff,
    (*st).do_dx, (*st).do_dy, (*st).dd_dx, (*st).dd_dy, (*st).cone_w, (*st).cone_spread
  );
  let flags = material_flags((*st).cand.material);
  let front = (*st).cand.toward > 0.0;
  var through = !front && (flags & MATERIAL_DOUBLE_SIDED) == 0u;
  if (camera && params.cull == CULL_BACK) {
    through = through || !front;
  } else if (camera && params.cull == CULL_FRONT) {
    through = through || front;
  }
  if (through) {
    continue_ray(st);
    return;
  }
  if ((flags & MATERIAL_MASK) != 0u) {
    request_eval(st, (*st).cand.material, surface_input((*st).cand), (*st).cand.has_uvs, true);
    (*st).phase = PH_ALPHA;
    return;
  }
  on_accept(st);
}

fn on_alpha(st: ptr<function, State>) {
  let cutoff = wordf(material_base((*st).cand.material) + 5u);
  if ((*st).sv.alpha < cutoff) {
    continue_ray(st);
  } else {
    on_accept(st);
  }
}

fn mis_weight_emission(st: ptr<function, State>) -> f32 {
  if (params.emitter_count == 0u) {
    return 1.0;
  }
  let pdf_light = emitter_pdf((*st).it);
  if (pdf_light > 0.0) {
    return power_heuristic((*st).prev_pdf, pdf_light);
  }
  return 1.0;
}

fn on_shade(st: ptr<function, State>) {
  let it = (*st).it;
  let sv = (*st).sv;
  if ((*st).vertex == 0u && (*st).first_t < 0.0) {
    (*st).first_t = length(it.p);  // (camera-relative: the distance from the eye)
  }
  let mb = material_base(it.material);
  let specular_factor = wordf(mb + 8u);
  var metallic = 0.0;
  var roughness = 1.0;
  var specular = 0.0;
  var has_specular = false;
  if ((material_flags(it.material) & MATERIAL_STANDARD) != 0u) {
    metallic = sv.metallic;
    roughness = sv.roughness;
    specular = specular_factor;
    has_specular = true;
  } else if (specular_factor > 0.0) {
    has_specular = true;
    specular = specular_factor;
    roughness = pow(2.0 / (wordf(mb + 9u) + 2.0), 0.25);
  }
  let direct = (*st).vertex <= 1u;
  if (any(sv.emissive != vec3<f32>(0.0))) {
    var le = sv.emissive;
    if ((*st).vertex > 0u) {
      le = le * mis_weight_emission(st);
    }
    add_target(st, (*st).throughput * le, direct);
  }
  var n_shade = sv.normal * it.toward;
  let ng_side = it.ng * it.toward;
  let wo = -it.d;
  if ((*st).vertex == 0u) {
    (*st).normal = n_shade;
  }
  if ((*st).vertex == params.bounces + 1u) {
    (*st).phase = PH_DONE;
    return;
  }
  if (dot(n_shade, wo) < MIN_COS) {
    n_shade = ng_side;
  }
  (*st).bsdf = bsdf_build(n_shade, wo, sv.base, metallic, roughness, specular, has_specular);
  if ((*st).vertex == 0u) {
    (*st).albedo = bsdf_albedo((*st).bsdf);
  }
  (*st).ng_side = ng_side;
  (*st).leave = offset_origin(it.p, ng_side);
  (*st).nee_light = 0u;
  (*st).nee_k = 0u;
  (*st).nee_index = 0u;
  (*st).nee_sum = vec3<f32>(0.0);
  (*st).phase = PH_NEE;
}

fn shadow_ray(st: ptr<function, State>, contrib: vec3<f32>, d: vec3<f32>, tmax: f32) {
  (*st).pending = contrib;
  start_ray(st, RAY_SHADOW, (*st).leave, d, 0.0, tmax, (*st).it.inst, (*st).it.prim);
  (*st).tally.shadow += 1u;
}

// The light sample of an emitter, once its emitted radiance `le` is known.
fn finish_emitter_sample(st: ptr<function, State>, le: vec3<f32>) -> bool {
  var le_over_pdf = vec3<f32>(0.0);
  if ((*st).em_ok) {
    le_over_pdf = le / (*st).em_pdf;
  }
  let wi = (*st).em_wi;
  let ev = bsdf_eval((*st).bsdf, wi);
  let ok = ev.cos > 0.0 && (*st).em_pdf > 0.0 && dot(wi, (*st).ng_side) > 0.0;
  let w = power_heuristic((*st).em_pdf, ev.pdf);
  let contrib = ev.f * le_over_pdf * (ev.cos * w / f32(params.light_samples));
  if (ok && any(contrib > vec3<f32>(0.0))) {
    shadow_ray(st, contrib, (*st).em_x - (*st).leave, SHADOW_END);
    return true;
  }
  return false;
}

fn on_nee(st: ptr<function, State>) {
  let ns = params.light_samples;
  let it = (*st).it;
  let lights = params.light_count + select(0u, 1u, env_sampled());  // the image after them
  loop {
    if ((*st).nee_light < lights) {
      if ((*st).nee_k >= ns) {
        (*st).nee_light += 1u;
        (*st).nee_k = 0u;
        continue;
      }
      let index = (*st).nee_index;
      (*st).nee_index += 1u;
      (*st).nee_k += 1u;
      if ((*st).nee_light == params.light_count) {  // the image environment
        let slot = SLOT_LIGHTS + 3u * index;
        let es = env_sample(rnd((*st).key, (*st).vertex, slot), rnd((*st).key, (*st).vertex, slot + 1u));
        let ev = bsdf_eval((*st).bsdf, es.d);
        let ok = ev.cos > 0.0 && es.pdf > 0.0 && dot(es.d, (*st).ng_side) > 0.0;
        if (ok) {
          let w = power_heuristic(es.pdf, ev.pdf);
          let contrib = ev.f * env_radiance(es.d) * (ev.cos * w / (es.pdf * f32(ns)));
          if (any(contrib > vec3<f32>(0.0))) {
            shadow_ray(st, contrib, es.d, PT_INF);
            return;
          }
        }
        continue;
      }
      let ls = sample_analytic((*st).nee_light, it.p, (*st).key, (*st).vertex, index);
      let ev = bsdf_eval((*st).bsdf, ls.wi);
      let ok = ev.cos > 0.0 && dot(ls.wi, (*st).ng_side) > 0.0 && !unlinked(it.inst, (*st).nee_light);
      var w = 1.0;
      if (!ls.delta) {
        w = power_heuristic(ls.pdf, ev.pdf);
      }
      let contrib = ev.f * ls.scale * (ev.cos * w / f32(ns));
      if (ok && any(contrib > vec3<f32>(0.0))) {
        if (pt[light_base((*st).nee_light) + 22u] != 0u) {
          (*st).nee_sum += contrib;  // casts no shadow: never occluded
          continue;
        }
        if (ls.has_target) {
          shadow_ray(st, contrib, ls.goal - (*st).leave, SHADOW_END);
        } else {
          shadow_ray(st, contrib, ls.wi, PT_INF);
        }
        return;
      }
      continue;
    }
    if (params.emitter_count > 0u && (*st).nee_k < ns) {
      let index = (*st).nee_index;
      (*st).nee_index += 1u;
      (*st).nee_k += 1u;
      let slot = SLOT_LIGHTS + 3u * index;
      let key = (*st).key;
      let k = pick_emitter(rnd(key, (*st).vertex, slot));
      let bary = uniform_triangle(rnd(key, (*st).vertex, slot + 1u), rnd(key, (*st).vertex, slot + 2u));
      let eb = emitter_base(k);
      let inst = pt[eb];
      let mesh_tri = pt[inst_base(inst) + 19u] + pt[eb + 1u];
      let c = triangle_corners(inst, mesh_tri);
      let x = bary.x * c.p0 + bary.y * c.p1 + bary.z * c.p2;
      let v = x - it.p;
      let dist2 = dot(v, v);
      let dist = sqrt(dist2);
      let wi = v / max(dist, 1e-30);
      let normal = ptv3(eb + 4u);
      let cos_l = -dot(normal, wi);
      let emits = cos_l > 0.0 || pt[eb + 3u] != 0u;
      let abs_cos = abs(cos_l);
      let ok = emits && abs_cos > 1e-12 && dist > 0.0;
      (*st).em_ok = ok;
      (*st).em_pdf = select(0.0, ptf(eb + 8u) / ptf(eb + 7u) * dist2 / abs_cos, ok);
      (*st).em_wi = wi;
      (*st).em_x = x;
      let m = pt[eb + 2u];
      let mflags = material_flags(m);
      if ((mflags & (MATERIAL_TEXTURED | MATERIAL_MASK)) != 0u) {
        // Textured or masked emission: evaluate the material at the sampled point.
        let toward = select(-1.0, 1.0, cos_l > 0.0);
        let rec = instances[inst];
        let has_uvs = (rec.flags & INSTANCE_HAS_UVS) != 0u;
        var uv = array<vec2<f32>, 3>(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
        if (has_uvs) {
          let tv = triangle_vertices(mesh_tri);
          uv = array<vec2<f32>, 3>(mesh_uv(tv.x), mesh_uv(tv.y), mesh_uv(tv.z));
        }
        let s = SurfaceInput(inst, bary, vec3<f32>(0.0), vec3<f32>(0.0), c.p0, c.p1, c.p2, uv[0], uv[1], uv[2], normal, normal, vec4<f32>(0.0), vertex_color(rec, triangle_vertices(mesh_tri), bary));
        (*st).em_masked = (mflags & MATERIAL_MASK) != 0u;
        (*st).em_cutoff = wordf(material_base(m) + 5u);
        request_eval(st, m, s, has_uvs, false);
        (*st).phase = PH_EMIT;
        return;
      }
      if (finish_emitter_sample(st, word3(material_base(m) + 10u))) {
        return;
      }
      continue;
    }
    // Every light sample is done: add them, then sample the BSDF.
    add_target(st, (*st).throughput * (*st).nee_sum, (*st).vertex == 0u);
    (*st).phase = PH_BOUNCE;
    return;
  }
}

fn on_emit(st: ptr<function, State>) {
  var le = (*st).sv.emissive;
  if ((*st).em_masked && !((*st).sv.alpha >= (*st).em_cutoff)) {
    le = vec3<f32>(0.0);
  }
  if (!finish_emitter_sample(st, le)) {
    (*st).phase = PH_NEE;
  }
}

fn on_bounce(st: ptr<function, State>) {
  let key = (*st).key;
  let vertex = (*st).vertex;
  let s = bsdf_sample(
    (*st).bsdf,
    rnd(key, vertex, SLOT_LOBE),
    rnd(key, vertex, SLOT_BSDF),
    rnd(key, vertex, SLOT_BSDF + 1u)
  );
  var valid = s.valid && dot(s.wi, (*st).ng_side) > 0.0;
  (*st).throughput = (*st).throughput * s.weight;
  if (vertex >= params.rr_vertex) {
    let q = clamp(max3((*st).throughput), RR_MIN, 1.0);
    valid = valid && rnd(key, vertex, SLOT_RR) < q;
    (*st).throughput = (*st).throughput / q;
  }
  if (!valid) {
    (*st).phase = PH_DONE;
    return;
  }
  (*st).prev_pdf = s.pdf;
  (*st).spread += select(DIFFUSE_SPREAD, 2.0 * (*st).bsdf.alpha, s.specular);
  start_ray(st, RAY_BOUNCE, (*st).leave, s.wi, 0.0, PT_INF, (*st).it.inst, (*st).it.prim);
  (*st).cone_w = (*st).it.width;
  (*st).cone_spread = (*st).spread;
  (*st).tally.bounce += 1u;
  (*st).vertex = vertex + 1u;
}

fn on_ao(st: ptr<function, State>) {
  let it = (*st).it;
  let ng_side = it.ng * it.toward;
  if ((*st).ao_j < params.ao_samples) {
    let j = (*st).ao_j;
    (*st).ao_j = j + 1u;
    let u1 = rnd((*st).key, 0u, SLOT_AO + 2u * j);
    let u2 = rnd((*st).key, 0u, SLOT_AO + 2u * j + 1u);
    let d = to_world(cosine_hemisphere(u1, u2), basis(ng_side), ng_side);
    start_ray(st, RAY_AO, offset_origin(it.p, ng_side), d, 0.0, params.ao_distance, it.inst, it.prim);
    (*st).tally.ao += 1u;
    return;
  }
  (*st).ao = 1.0 - params.ao_strength * (*st).ao_blocked / f32(params.ao_samples);
  if (params.integrator == INTEGRATOR_AO) {
    (*st).direct = vec3<f32>((*st).ao);
    (*st).phase = PH_DONE;
    return;
  }
  request_eval(st, it.material, surface_input(it), it.has_uvs, false);
  (*st).phase = PH_SHADE;
}

fn run_state(st: ptr<function, State>) {
  loop {
    if ((*st).phase == PH_TRACE) {
      let shadow = (*st).kind == RAY_SHADOW || (*st).kind == RAY_AO;
      (*st).hit = trace_ray(
        (*st).ro, (*st).rd, (*st).rtmin, (*st).rtmax, (*st).skip_inst, (*st).skip_prim,
        shadow && !scene_has_pass_through()
      );
      (*st).phase = PH_HIT;
    }
    if ((*st).eval_pending) {
      // The one place materials are evaluated (see the header).
      (*st).sv = evaluate_surface((*st).eval_m, (*st).eval_in, (*st).eval_uvs, (*st).eval_alpha);
      (*st).eval_pending = false;
    }
    // An if-chain, not a switch: D3D12's FXC cannot inline these calls inside the cases
    // of a switch in this loop ("internal error: l-value expected").
    let phase = (*st).phase;
    if (phase == PH_HIT) {
      on_hit(st);
    } else if (phase == PH_ALPHA) {
      on_alpha(st);
    } else if (phase == PH_SHADE) {
      on_shade(st);
    } else if (phase == PH_NEE) {
      on_nee(st);
    } else if (phase == PH_EMIT) {
      on_emit(st);
    } else if (phase == PH_BOUNCE) {
      on_bounce(st);
    } else if (phase == PH_AO) {
      on_ao(st);
    }
    if ((*st).phase == PH_DONE) {
      break;
    }
  }
}

fn new_state(key: vec2<u32>) -> State {
  var st: State;
  st.key = key;
  st.phase = PH_DONE;
  st.eval_pending = false;
  st.vertex = 0u;
  st.throughput = vec3<f32>(1.0);
  st.prev_pdf = 0.0;
  st.spread = 0.0;
  st.direct = vec3<f32>(0.0);
  st.indirect = vec3<f32>(0.0);
  st.albedo = vec3<f32>(0.0);
  st.normal = vec3<f32>(0.0);
  st.ao = 1.0;
  st.hit_any = false;
  st.first_t = -1.0;
  st.depth = PT_INF;
  st.center_inst = PT_NONE;
  st.tally = Tally(0u, 0u, 0u, 0u, 0u, 0u, 0u);
  return st;
}

fn add_count(k: u32, n: u32) {
  if (n != 0u) {
    atomicAdd(&wg_counts[k], n);
  }
}

fn plane(k: u32, pixel: u32) -> u32 {
  return k * params.n_pix + pixel;
}

@compute @workgroup_size(8, 8)
fn path_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  if (lid < 8u) {
    atomicStore(&wg_counts[lid], 0u);
  }
  workgroupBarrier();
  var tally = Tally(0u, 0u, 0u, 0u, 0u, 0u, 0u);
  let x = params.tile_x + gid.x;
  let y = params.tile_y + gid.y;
  if (gid.x < params.tile_w && gid.y < params.tile_h) {
    let pixel = y * params.width + x;
    if (params.mode == MODE_RNG) {
      // Test hook: the random numbers of a few decisions (compared with the CPU's).
      let key = path_key(pixel, params.sample_first);
      accum[plane(0u, pixel)] = rnd(key, 0u, SLOT_PIXEL);
      accum[plane(1u, pixel)] = rnd(key, 0u, SLOT_PIXEL + 1u);
      accum[plane(2u, pixel)] = rnd(key, 3u, SLOT_LIGHTS + 5u);
      accum[plane(3u, pixel)] = rnd(key, 2u, SLOT_RR);
    } else if (params.mode == MODE_CENTER) {
      var st = new_state(vec2<u32>(0u));
      camera_ray(&st, RAY_CENTER, x, y, 0.5, 0.5, false, 0.5, 0.5);
      st.tally.center += 1u;
      run_state(&st);
      accum[plane(ACC_DEPTH, pixel)] = st.depth;
      accum[plane(ACC_INSTANCE, pixel)] = bitcast<f32>(st.center_inst);
      tally = st.tally;
    } else {
      for (var s = params.sample_first; s < params.sample_first + params.sample_count; s++) {
        let key = path_key(pixel, s);
        var st = new_state(key);
        var lu = 0.5;
        var lv = 0.5;
        if (params.lens.z > 0.0) {
          lu = rnd(key, 0u, SLOT_LENS);
          lv = rnd(key, 0u, SLOT_LENS + 1u);
        }
        camera_ray(&st, RAY_CAMERA, x, y, rnd(key, 0u, SLOT_PIXEL), rnd(key, 0u, SLOT_PIXEL + 1u), true, lu, lv);
        st.cam_d = st.rd;
        st.tally.camera += 1u;
        run_state(&st);
        if (params.atmo.fog.w > 0.5) {  // the camera segment through the fog
          let trans = atmo_transmittance(params.atmo, st.cam_d, st.first_t);
          st.direct = atmo_fog(params.atmo, st.direct, st.cam_d, st.first_t);
          st.indirect = st.indirect * trans;
        }
        var indirect = st.indirect;
        if (params.clamp > 0.0) {
          let peak = max3(indirect);
          if (peak > params.clamp) {
            indirect = indirect * (params.clamp / peak);
          }
        }
        let beauty = st.direct + indirect;
        for (var c = 0u; c < 3u; c++) {
          accum[plane(ACC_DIRECT + c, pixel)] += st.direct[c];
          accum[plane(ACC_INDIRECT + c, pixel)] += indirect[c];
          accum[plane(ACC_ALBEDO + c, pixel)] += st.albedo[c];
          accum[plane(ACC_NORMAL + c, pixel)] += st.normal[c];
          // Welford's running mean and sum of squared deviations (the variance pass).
          let n = f32(s + 1u);
          let mean = accum[plane(ACC_MEAN + c, pixel)];
          let delta = beauty[c] - mean;
          let next = mean + delta / n;
          accum[plane(ACC_MEAN + c, pixel)] = next;
          accum[plane(ACC_M2 + c, pixel)] += delta * (beauty[c] - next);
        }
        accum[plane(ACC_AO, pixel)] += st.ao;
        accum[plane(ACC_HITS, pixel)] += select(0.0, 1.0, st.hit_any);
        tally = add_tally(tally, st.tally);
      }
    }
  }
  add_count(C_CAMERA, tally.camera);
  add_count(C_BOUNCE, tally.bounce);
  add_count(C_SHADOW, tally.shadow);
  add_count(C_AO, tally.ao);
  add_count(C_CENTER, tally.center);
  add_count(C_THROUGH, tally.through);
  add_count(C_LIMITED, tally.limited);
  workgroupBarrier();
  if (lid < 8u) {
    let total = atomicLoad(&wg_counts[lid]);
    if (total != 0u) {
      atomicAdd(&counters[lid], total);
    }
  }
}
