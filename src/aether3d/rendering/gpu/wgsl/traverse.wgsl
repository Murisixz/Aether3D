// Two-level BVH traversal of the CPU path tracer's acceleration structure
// (rendering/raytrace/accel.py), shared by the GPU path tracer (pathtrace.wgsl) and the GPU
// bake kernel (bake.wgsl). The including module declares the scene words
// `pt: array<u32>` and the functions trace_root_lo(), trace_root_hi() and
// trace_instance_count().

const PT_NONE: u32 = 0xffffffffu;
const PT_STACK: u32 = 64u;
const PT_INF: f32 = 3.0e38;
const MIN_DIRECTION: f32 = 1e-30;
const BOX_PAD: f32 = 1.0000004;     // 1 + 2 gamma(3): conservative float32 slab tests
const KIND_INNER: u32 = 0u;
const KIND_TRIANGLES: u32 = 1u;
const KIND_INSTANCES: u32 = 2u;

// Header of the path tracer's data words (pt): where each region starts.
const H_NODES: u32 = 0u;     // 4 words per node: a, b, kind, -
const H_BOXES: u32 = 1u;     // 12 floats per node: both children's lo/hi
const H_TRIS: u32 = 2u;      // 9 floats per leaf-order triangle (object space)
const H_PRIM: u32 = 3u;      // mesh triangle of each leaf-order triangle
const H_SLOTS: u32 = 4u;     // instance of each top-level leaf slot
const H_INST: u32 = 5u;      // INST_WORDS per instance
const H_LIGHTS: u32 = 6u;    // LIGHT_WORDS per analytic light
const H_EMITTERS: u32 = 7u;  // EMITTER_WORDS per emissive triangle
const H_CDF: u32 = 8u;       // cumulative selection probability per emitter
const H_LOOKUP: u32 = 9u;    // global triangle -> emitter (PT_NONE: none)
const H_ENV: u32 = 10u;      // image environment: W, H, nu, nv, pixels, marginal, conditional
const H_SOBOL: u32 = 11u;    // the Sobol' generator matrices: 4 x 32 columns (sampling.py)
const H_FILTER: u32 = 12u;   // the pixel filter's inverse CDF (FILTER_TABLE floats)
const INST_WORDS: u32 = 24u;

fn ptf(i: u32) -> f32 {
  return bitcast<f32>(pt[i]);
}

fn ptv3(i: u32) -> vec3<f32> {
  return vec3<f32>(ptf(i), ptf(i + 1u), ptf(i + 2u));
}

// -- random numbers: SplitMix64 in 32-bit pairs (lo, hi) -----------------------------------

const GOLDEN: vec2<u32> = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
const MIX_1: vec2<u32> = vec2<u32>(0x1ce4e5b9u, 0xbf58476du);
const MIX_2: vec2<u32> = vec2<u32>(0x133111ebu, 0x94d049bbu);
const PIXEL_MUL: vec2<u32> = vec2<u32>(0xd192ed03u, 0xd1b54a32u);
const SAMPLE_MUL: vec2<u32> = vec2<u32>(0x108ef2d9u, 0xaef17502u);
const DIM_MUL: vec2<u32> = vec2<u32>(0x75ae2165u, 0xdb4f0b91u);

// -- the BVH: two-level traversal (rendering/raytrace/accel.py) -----------------------------

struct Hit {
  t: f32,
  inst: u32,   // PT_NONE: no hit
  prim: u32,   // mesh triangle
  b: vec3<f32>,
};

struct RaySetup {
  inv: vec3<f32>,
  k: vec3<u32>,      // kx, ky, kz of the watertight test
  shear: vec3<f32>,
};

fn ray_setup(d: vec3<f32>) -> RaySetup {
  let tiny = abs(d) < vec3<f32>(MIN_DIRECTION);
  let signed_tiny = select(vec3<f32>(MIN_DIRECTION), vec3<f32>(-MIN_DIRECTION), d < vec3<f32>(0.0));
  let inv = 1.0 / select(d, signed_tiny, tiny);
  let a = abs(d);
  var kz = 0u;
  if (a.y > a.x) {
    kz = 1u;
  }
  if (a.z > a[kz]) {
    kz = 2u;
  }
  var kx = (kz + 1u) % 3u;
  var ky = (kx + 1u) % 3u;
  let dz = d[kz];
  if (dz < 0.0) {
    let swap = kx;
    kx = ky;
    ky = swap;
  }
  let dz_safe = select(dz, 1.0, dz == 0.0);
  return RaySetup(inv, vec3<u32>(kx, ky, kz), vec3<f32>(d[kx] / dz_safe, d[ky] / dz_safe, 1.0 / dz_safe));
}

// Slab test: (entry distance, 1 if the box is hit in range else 0).
fn box_hit(lo: vec3<f32>, hi: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, tmin: f32, tmax: f32) -> vec2<f32> {
  let t1 = (lo - o) * inv;
  let t2 = (hi - o) * inv;
  let a = min(t1, t2);
  let b = max(t1, t2);
  let near = max(max(a.x, a.y), max(a.z, tmin));
  let far = min(min(b.x, b.y), b.z) * BOX_PAD;
  return vec2<f32>(near, select(0.0, 1.0, near <= far && near < tmax));
}

struct TriHit {
  t: f32,
  b: vec3<f32>,
  ok: bool,
};

fn tri_hit(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, o: vec3<f32>, rs: RaySetup) -> TriHit {
  let a = v0 - o;
  let b = v1 - o;
  let c = v2 - o;
  let kx = rs.k.x;
  let ky = rs.k.y;
  let kz = rs.k.z;
  let az = a[kz];
  let bz = b[kz];
  let cz = c[kz];
  let ax = a[kx] - rs.shear.x * az;
  let ay = a[ky] - rs.shear.y * az;
  let bx = b[kx] - rs.shear.x * bz;
  let by = b[ky] - rs.shear.y * bz;
  let cx = c[kx] - rs.shear.x * cz;
  let cy = c[ky] - rs.shear.y * cz;
  let u = cx * by - cy * bx;
  let v = ax * cy - ay * cx;
  let w = bx * ay - by * ax;
  let det = u + v + w;
  let ok = !((u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0)) && det != 0.0;
  let safe = select(1.0, det, det != 0.0);
  let sz = rs.shear.z;
  let t = (u * (sz * az) + v * (sz * bz) + w * (sz * cz)) / safe;
  return TriHit(t, vec3<f32>(u, v, w) / safe, ok);
}

fn inst_base(inst: u32) -> u32 {
  return pt[H_INST] + inst * INST_WORDS;
}

fn xform_point(ib: u32, p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    dot(ptv3(ib), p) + ptf(ib + 3u),
    dot(ptv3(ib + 4u), p) + ptf(ib + 7u),
    dot(ptv3(ib + 8u), p) + ptf(ib + 11u)
  );
}

fn xform_direction(ib: u32, d: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(ptv3(ib), d), dot(ptv3(ib + 4u), d), dot(ptv3(ib + 8u), d));
}

// Closest hit in (tmin, tmax) -- or any hit -- skipping the triangle a ray leaves from.
fn trace_ray(o_w: vec3<f32>, d_w: vec3<f32>, tmin: f32, tmax: f32, skip_inst: u32, skip_prim: u32, any_hit: bool) -> Hit {
  var best = Hit(tmax, PT_NONE, PT_NONE, vec3<f32>(0.0));
  var t_best = tmax;
  let world = ray_setup(d_w);
  let root = box_hit(trace_root_lo(), trace_root_hi(), o_w, world.inv, tmin, t_best);
  if (root.y == 0.0 || trace_instance_count() == 0u) {
    return best;
  }
  var stack_node: array<u32, PT_STACK>;
  var stack_tag: array<u32, PT_STACK>;
  var stack_t: array<f32, PT_STACK>;
  var sp = 0u;
  var node = 0u;
  var tag = 0u;          // 0: the top level (world); k + 1: inside instance k
  var t_node = root.x;
  var cur = 0u;
  var o = o_w;
  var rs = world;
  loop {
    var pop = true;
    if (t_node < t_best) {
      if (tag != cur) {
        if (tag == 0u) {
          o = o_w;
          rs = world;
        } else {
          let ib = inst_base(tag - 1u);
          o = xform_point(ib, o_w);
          rs = ray_setup(xform_direction(ib, d_w));
        }
        cur = tag;
      }
      let nb = pt[H_NODES] + 4u * node;
      let kind = pt[nb + 2u];
      if (kind == KIND_INNER) {
        let bb = pt[H_BOXES] + 12u * node;
        let h0 = box_hit(ptv3(bb), ptv3(bb + 3u), o, rs.inv, tmin, t_best);
        let h1 = box_hit(ptv3(bb + 6u), ptv3(bb + 9u), o, rs.inv, tmin, t_best);
        let ok0 = h0.y > 0.5;
        let ok1 = h1.y > 0.5;
        if (ok0 || ok1) {
          let c0 = pt[nb];
          let c1 = pt[nb + 1u];
          let near0 = select(ok0, h0.x <= h1.x, ok0 && ok1);
          if (ok0 && ok1 && sp < PT_STACK) {
            stack_node[sp] = select(c0, c1, near0);
            stack_tag[sp] = tag;
            stack_t[sp] = select(h0.x, h1.x, near0);
            sp += 1u;
          }
          node = select(c1, c0, near0);
          t_node = select(h1.x, h0.x, near0);
          pop = false;
        }
      } else if (kind == KIND_INSTANCES) {
        let first = pt[nb];
        let count = pt[nb + 1u];
        for (var k = 0u; k < count; k++) {
          let ins = pt[pt[H_SLOTS] + first + k];
          let ib = inst_base(ins);
          let h = box_hit(ptv3(ib + 12u), ptv3(ib + 15u), o_w, world.inv, tmin, t_best);
          if (h.y > 0.5 && sp < PT_STACK) {
            stack_node[sp] = pt[ib + 18u];
            stack_tag[sp] = ins + 1u;
            stack_t[sp] = h.x;
            sp += 1u;
          }
        }
      } else {
        let first = pt[nb];
        let count = pt[nb + 1u];
        let ins = tag - 1u;
        for (var k = 0u; k < count; k++) {
          let g = first + k;
          let tb = pt[H_TRIS] + 9u * g;
          let th = tri_hit(ptv3(tb), ptv3(tb + 3u), ptv3(tb + 6u), o, rs);
          if (th.ok && th.t > tmin && th.t < t_best) {
            let prim = pt[pt[H_PRIM] + g];
            if (!(ins == skip_inst && prim == skip_prim)) {
              t_best = th.t;
              best = Hit(th.t, ins, prim, th.b);
              if (any_hit) {
                return best;
              }
            }
          }
        }
      }
    }
    if (pop) {
      if (sp == 0u) {
        break;
      }
      sp -= 1u;
      node = stack_node[sp];
      tag = stack_tag[sp];
      t_node = stack_t[sp];
    }
  }
  return best;
}

