// Aether3D GPU renderer -- materials (appended to common.wgsl; used by the scene and shadow passes).
//
// Everything a material needs lives in two storage buffers built on the CPU
// (aether3d/rendering/gpu/materials.py) -- never in shader code:
//
//   material_data: u32 words -- a header, then per-instance material tables,
//                  material records, texture bindings, compiled texture
//                  programs, their parameters (f32 bits), image descriptors and
//                  the sRGB decoding table;
//   texels:        every image's mip levels, one RGBA8 texel per u32.
//
// Procedural textures are compiled programs (aether3d/materials/graph.py) run
// by the interpreter below; images are sampled manually (wrap, bilinear,
// trilinear) so the GPU computes what the CPU renderer computes
// (aether3d/materials/sampling.py, aether3d/rendering/surface.py). Hashes and
// noise use the same integer arithmetic as the CPU, bit for bit.

// -- layout constants (mirrored in materials.py; a test checks they agree) --------------
const HEADER_INSTANCES: u32 = 0u;
const HEADER_MATERIALS: u32 = 1u;
const HEADER_BINDINGS: u32 = 2u;
const HEADER_PROGRAMS: u32 = 3u;
const HEADER_PARAMS: u32 = 4u;
const HEADER_IMAGES: u32 = 5u;
const HEADER_LUT: u32 = 6u;
const HEADER_GROUPS: u32 = 7u;
const INSTANCE_WORDS: u32 = 4u;
const MATERIAL_WORDS: u32 = 24u;
const BINDING_WORDS: u32 = 20u;
const IMAGE_WORDS: u32 = 8u;
const INSTRUCTION_WORDS: u32 = 8u;
const NO_BINDING: u32 = 0xffffffffu;

const MATERIAL_STANDARD: u32 = 1u;
const MATERIAL_MASK: u32 = 2u;
const MATERIAL_DOUBLE_SIDED: u32 = 4u;
const MATERIAL_TEXTURED: u32 = 8u;

const SLOT_BASE_COLOR: u32 = 0u;
const SLOT_OPACITY: u32 = 1u;
const SLOT_METALLIC: u32 = 2u;
const SLOT_ROUGHNESS: u32 = 3u;
const SLOT_NORMAL: u32 = 4u;
const SLOT_EMISSIVE: u32 = 5u;
const SLOT_OCCLUSION: u32 = 6u;

const COORD_UV: u32 = 0u;
const COORD_OBJECT: u32 = 1u;
const COORD_WORLD: u32 = 2u;
const PROJ_TRIPLANAR: u32 = 0u;
const PROJ_XY: u32 = 1u;
const PROJ_XZ: u32 = 2u;
const PROJ_YZ: u32 = 3u;
const PROJ_VOLUME: u32 = 4u;
const BINDING_IMAGE: u32 = 4096u;  // bit 12: the texture is an image (its alpha is opacity)
const BINDING_MIKK: u32 = 8192u;  // bit 13: a UV normal map read in the MikkTSpace basis

const TRIPLANAR_EPSILON: f32 = 1e-4;
const MIN_ROUGHNESS: f32 = 0.045;
const PI: f32 = 3.14159265358979;

@group(0) @binding(6) var<storage, read> material_data: array<u32>;
@group(0) @binding(7) var<storage, read> texels: array<u32>;

fn word(i: u32) -> u32 {
  return material_data[i];
}

fn wordf(i: u32) -> f32 {
  return bitcast<f32>(material_data[i]);
}

fn word3(i: u32) -> vec3<f32> {
  return vec3<f32>(wordf(i), wordf(i + 1u), wordf(i + 2u));
}

fn header(k: u32) -> u32 {
  return material_data[k];
}

fn material_base(m: u32) -> u32 {
  return header(HEADER_MATERIALS) + m * MATERIAL_WORDS;
}

// The material of a triangle of a drawn instance (default, or its face group's).
fn triangle_material(instance: u32, group: u32) -> u32 {
  let t = header(HEADER_INSTANCES) + instance * INSTANCE_WORDS;
  let count = word(t + 2u);
  if (group < count) {
    return word(header(HEADER_GROUPS) + word(t + 1u) + group);
  }
  return word(t);
}

// -- the vertex stage shared by the scene and the masked shadow pass -------------------------

struct VertexOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) @interpolate(perspective, centroid) position: vec3<f32>,  // camera-relative world
  @location(1) @interpolate(perspective, centroid) normal: vec3<f32>,    // world, unit (or zero)
  @location(2) @interpolate(flat) face: vec3<f32>,                       // world face normal (outward)
  @location(3) @interpolate(flat) instance: u32,
  @location(4) @interpolate(perspective, centroid) barycentric: vec3<f32>,
  // The same barycentrics at the pixel center, for screen-space derivatives: derivatives of
  // centroid-interpolated values are unreliable along edges (the centroid moves with the
  // coverage, and helper lanes have none), and they choose mip levels and bump steps.
  @location(14) @interpolate(perspective) center: vec3<f32>,
  // The whole triangle (camera-relative corners and their world normals), for shadow lookups.
  @location(5) @interpolate(flat) p0: vec3<f32>,
  @location(6) @interpolate(flat) p1: vec3<f32>,
  @location(7) @interpolate(flat) p2: vec3<f32>,
  @location(8) @interpolate(flat) n0: vec3<f32>,
  @location(9) @interpolate(flat) n1: vec3<f32>,
  @location(10) @interpolate(flat) n2: vec3<f32>,
  // Texture coordinates of the corners (zero without UVs) and the triangle's material.
  @location(11) @interpolate(flat) uv01: vec4<f32>,
  @location(12) @interpolate(flat) uv2: vec2<f32>,
  @location(13) @interpolate(flat) material: u32,
  // The interpolated MikkTSpace tangent (world, not renormalized) and its sign, for UV normal
  // maps (zero otherwise).
  @location(15) @interpolate(perspective, centroid) tangent: vec4<f32>,
  // The interpolated vertex colour (linear RGBA; white without colours): multiplies the
  // material's base colour and opacity, as glTF's COLOR_0.
  @location(16) @interpolate(perspective, centroid) color: vec4<f32>,
};

fn build_vertex(vertex: u32, instance: u32, view_proj: mat4x4<f32>) -> VertexOut {
  let inst = instances[instance];
  let triangle = vertex / 3u;
  let tri = triangle_vertices(triangle);
  let q0 = mesh_position(tri.x);
  let q1 = mesh_position(tri.y);
  let q2 = mesh_position(tri.z);
  let corner = vertex % 3u;
  var out: VertexOut;
  out.p0 = relative_position(inst, q0);
  out.p1 = relative_position(inst, q1);
  out.p2 = relative_position(inst, q2);
  var position = out.p0;
  out.barycentric = vec3<f32>(1.0, 0.0, 0.0);
  if (corner == 1u) {
    position = out.p1;
    out.barycentric = vec3<f32>(0.0, 1.0, 0.0);
  } else if (corner == 2u) {
    position = out.p2;
    out.barycentric = vec3<f32>(0.0, 0.0, 1.0);
  }
  out.center = out.barycentric;
  out.clip = view_proj * vec4<f32>(position, 1.0);
  out.position = position;
  // Object-space face normal through the normal matrix: outward, also for mirrored instances.
  out.face = safe_normalize(world_normal(inst, cross(q1 - q0, q2 - q0)));
  out.normal = vec3<f32>(0.0);
  out.n0 = vec3<f32>(0.0);
  out.n1 = vec3<f32>(0.0);
  out.n2 = vec3<f32>(0.0);
  if ((inst.flags & INSTANCE_HAS_NORMALS) != 0u) {
    out.n0 = safe_normalize(world_normal(inst, mesh_normal(tri.x)));
    out.n1 = safe_normalize(world_normal(inst, mesh_normal(tri.y)));
    out.n2 = safe_normalize(world_normal(inst, mesh_normal(tri.z)));
    out.normal = out.n0;
    if (corner == 1u) {
      out.normal = out.n1;
    } else if (corner == 2u) {
      out.normal = out.n2;
    }
  }
  out.uv01 = vec4<f32>(0.0);
  out.uv2 = vec2<f32>(0.0);
  if ((inst.flags & INSTANCE_HAS_UVS) != 0u) {
    out.uv01 = vec4<f32>(mesh_uv(tri.x), mesh_uv(tri.y));
    out.uv2 = mesh_uv(tri.z);
  }
  var group = 0u;
  if ((inst.flags & INSTANCE_HAS_GROUPS) != 0u) {
    group = mesh_groups[triangle];
  }
  out.material = triangle_material(instance, group);
  out.instance = instance;
  out.tangent = vec4<f32>(0.0);
  if ((inst.flags & INSTANCE_HAS_UVS) != 0u && needs_tangents(out.material)) {
    out.tangent = world_tangent(inst, mesh_tangent(3u * triangle + corner));
  }
  out.color = vec4<f32>(1.0);
  if ((inst.flags & INSTANCE_HAS_COLORS) != 0u) {
    out.color = mesh_color(inst, tri[corner]);
  }
  return out;
}

// -- hashes and noise (aether3d/materials/noise.py) -----------------------------------------

const COORD_LIMIT: f32 = 1.0e6;
const SEED_MIX: u32 = 0x9e3779b9u;
const INV_2_24: f32 = 1.0 / 16777216.0;

fn pcg3d(v_in: vec3<u32>) -> vec3<u32> {
  var v = v_in * 1664525u + vec3<u32>(1013904223u);
  v.x = v.x + v.y * v.z;
  v.y = v.y + v.z * v.x;
  v.z = v.z + v.x * v.y;
  v = v ^ (v >> vec3<u32>(16u));
  v.x = v.x + v.y * v.z;
  v.y = v.y + v.z * v.x;
  v.z = v.z + v.x * v.y;
  return v;
}

fn unit(h: u32) -> f32 {
  return f32(h >> 8u) * INV_2_24;
}

fn seed_bits(seed: u32) -> u32 {
  return seed * SEED_MIX;
}

struct Cell {
  bits: vec3<u32>,
  frac: vec3<f32>,
};

fn cell_index(p: vec3<f32>) -> Cell {
  let c = clamp(p, vec3<f32>(-COORD_LIMIT), vec3<f32>(COORD_LIMIT));
  let w = floor(c);
  return Cell(bitcast<vec3<u32>>(vec3<i32>(w)), c - w);
}

fn fade(t: vec3<f32>) -> vec3<f32> {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn corner_hash(bits: vec3<u32>, d: vec3<u32>, seed: u32) -> u32 {
  return pcg3d(vec3<u32>(bits.x + d.x, bits.y + d.y, (bits.z + d.z) ^ seed)).x;
}

fn trilinear(c: array<f32, 8>, u: vec3<f32>) -> f32 {
  let x00 = c[0] + (c[1] - c[0]) * u.x;
  let x10 = c[2] + (c[3] - c[2]) * u.x;
  let x01 = c[4] + (c[5] - c[4]) * u.x;
  let x11 = c[6] + (c[7] - c[6]) * u.x;
  let y0 = x00 + (x10 - x00) * u.y;
  let y1 = x01 + (x11 - x01) * u.y;
  return y0 + (y1 - y0) * u.z;
}

fn gradient_of(h: u32) -> vec3<f32> {
  switch (h % 12u) {
    case 0u: { return vec3<f32>(1.0, 1.0, 0.0); }
    case 1u: { return vec3<f32>(-1.0, 1.0, 0.0); }
    case 2u: { return vec3<f32>(1.0, -1.0, 0.0); }
    case 3u: { return vec3<f32>(-1.0, -1.0, 0.0); }
    case 4u: { return vec3<f32>(1.0, 0.0, 1.0); }
    case 5u: { return vec3<f32>(-1.0, 0.0, 1.0); }
    case 6u: { return vec3<f32>(1.0, 0.0, -1.0); }
    case 7u: { return vec3<f32>(-1.0, 0.0, -1.0); }
    case 8u: { return vec3<f32>(0.0, 1.0, 1.0); }
    case 9u: { return vec3<f32>(0.0, -1.0, 1.0); }
    case 10u: { return vec3<f32>(0.0, 1.0, -1.0); }
    default: { return vec3<f32>(0.0, -1.0, -1.0); }
  }
}

// Value noise (kind 1) or gradient noise (kind 0), in [0, 1].
fn noise3(p: vec3<f32>, seed: u32, kind: u32) -> f32 {
  let cell = cell_index(p);
  let s = seed_bits(seed);
  let u = fade(cell.frac);
  var c: array<f32, 8>;
  for (var k = 0u; k < 8u; k++) {
    let d = vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
    let h = corner_hash(cell.bits, d, s);
    if (kind == 1u) {
      c[k] = unit(h);
    } else {
      c[k] = dot(gradient_of(h), cell.frac - vec3<f32>(d));
    }
  }
  let n = trilinear(c, u);
  if (kind == 1u) {
    return n;
  }
  return clamp(0.5 + 0.5 * n, 0.0, 1.0);
}

fn fbm(p: vec3<f32>, seed: u32, kind: u32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
  var total = 0.0;
  var amplitude = 1.0;
  var frequency = 1.0;
  var norm = 0.0;
  for (var octave = 0u; octave < octaves; octave++) {
    total = total + noise3(p * frequency, seed + octave, kind) * amplitude;
    norm = norm + amplitude;
    amplitude = amplitude * gain;
    frequency = frequency * lacunarity;
  }
  return total / norm;
}

fn voronoi(p_in: vec3<f32>, seed: u32, randomness: f32, dimensions: u32) -> vec3<f32> {
  var p = p_in;
  if (dimensions == 2u) {
    p.z = 0.0;
  }
  let cell = cell_index(p);
  let s = seed_bits(seed);
  var f1 = 8.0;
  var f2 = 8.0;
  var value = 0.0;
  var z0 = -1;
  var z1 = 1;
  if (dimensions == 2u) {
    z0 = 0;
    z1 = 0;
  }
  for (var dz = z0; dz <= z1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let d = bitcast<vec3<u32>>(vec3<i32>(dx, dy, dz));
        let h = pcg3d(vec3<u32>(cell.bits.x + d.x, cell.bits.y + d.y, (cell.bits.z + d.z) ^ s));
        let px = f32(dx) + 0.5 + (unit(h.x) - 0.5) * randomness - cell.frac.x;
        let py = f32(dy) + 0.5 + (unit(h.y) - 0.5) * randomness - cell.frac.y;
        var dist: f32;
        if (dimensions == 2u) {
          dist = sqrt(px * px + py * py);
        } else {
          let pz = f32(dz) + 0.5 + (unit(h.z) - 0.5) * randomness - cell.frac.z;
          dist = sqrt(px * px + py * py + pz * pz);
        }
        let v = unit(h.x + h.y + h.z);
        if (dist < f1) {
          f2 = f1;
          value = v;
          f1 = dist;
        } else {
          f2 = min(f2, dist);
        }
      }
    }
  }
  return vec3<f32>(clamp(f1, 0.0, 1.0), clamp(f2 - f1, 0.0, 1.0), value);
}

fn cell_random(ix: f32, iy: f32, seed: u32) -> f32 {
  let b = bitcast<vec2<u32>>(vec2<i32>(i32(ix), i32(iy)));
  return unit(pcg3d(vec3<u32>(b.x, b.y, 0u ^ seed_bits(seed))).x);
}

// smoothstep with numpy's rule for equal edges (a step at e1), unlike WGSL's built-in.
fn smooth_step(e0: f32, e1: f32, x: f32) -> f32 {
  let span = e1 - e0;
  if (span == 0.0) {
    return select(0.0, 1.0, x >= e1);
  }
  let t = clamp((x - e0) / span, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// -- images (aether3d/materials/sampling.py) ---------------------------------------------------

fn wrap_index(i: i32, size: i32, mode: u32) -> i32 {
  if (mode == 1u) {  // clamp
    return clamp(i, 0, size - 1);
  }
  if (mode == 2u) {  // mirror
    let period = 2 * size;
    let m = ((i % period) + period) % period;
    return select(period - 1 - m, m, m < size);
  }
  return ((i % size) + size) % size;  // repeat
}

fn fetch_texel(index: u32, srgb: bool) -> vec4<f32> {
  let w = texels[index];
  let r = w & 0xffu;
  let g = (w >> 8u) & 0xffu;
  let b = (w >> 16u) & 0xffu;
  let a = f32(w >> 24u) / 255.0;
  if (srgb) {
    let lut = header(HEADER_LUT);
    return vec4<f32>(wordf(lut + r), wordf(lut + g), wordf(lut + b), a);
  }
  return vec4<f32>(f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0, a);
}

struct ImageInfo {
  offset: u32,
  width: u32,
  height: u32,
  levels: u32,
  wrap_u: u32,
  wrap_v: u32,
  linear_filter: bool,
  srgb: bool,
  mipmaps: bool,
};

fn image_info(index: u32) -> ImageInfo {
  let b = header(HEADER_IMAGES) + index * IMAGE_WORDS;
  let flags = word(b + 4u);
  return ImageInfo(
    word(b), word(b + 1u), word(b + 2u), word(b + 3u),
    flags & 3u, (flags >> 2u) & 3u, (flags & 16u) != 0u, (flags & 32u) != 0u, (flags & 64u) != 0u
  );
}

fn sample_level(img: ImageInfo, level: u32, uv: vec2<f32>, linear_filter: bool) -> vec4<f32> {
  var offset = img.offset;
  var w = img.width;
  var h = img.height;
  for (var k = 0u; k < level; k++) {
    offset = offset + w * h;
    w = max(1u, w / 2u);
    h = max(1u, h / 2u);
  }
  let wi = i32(w);
  let hi = i32(h);
  let tx = uv.x * f32(w);
  let ty = (1.0 - uv.y) * f32(h);
  if (!linear_filter) {
    let x = wrap_index(i32(floor(tx)), wi, img.wrap_u);
    let y = wrap_index(i32(floor(ty)), hi, img.wrap_v);
    return fetch_texel(offset + u32(y * wi + x), img.srgb);
  }
  let fx = tx - 0.5;
  let fy = ty - 0.5;
  let x0f = floor(fx);
  let y0f = floor(fy);
  let ax = fx - x0f;
  let ay = fy - y0f;
  let xa = wrap_index(i32(x0f), wi, img.wrap_u);
  let xb = wrap_index(i32(x0f) + 1, wi, img.wrap_u);
  let ya = wrap_index(i32(y0f), hi, img.wrap_v);
  let yb = wrap_index(i32(y0f) + 1, hi, img.wrap_v);
  let top = fetch_texel(offset + u32(ya * wi + xa), img.srgb) * (1.0 - ax)
    + fetch_texel(offset + u32(ya * wi + xb), img.srgb) * ax;
  let bottom = fetch_texel(offset + u32(yb * wi + xa), img.srgb) * (1.0 - ax)
    + fetch_texel(offset + u32(yb * wi + xb), img.srgb) * ax;
  return top * (1.0 - ay) + bottom * ay;
}

fn sample_image(index: u32, p: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> vec4<f32> {
  let img = image_info(index);
  let uv = p.xy;
  if (!img.mipmaps || img.levels <= 1u) {
    return sample_level(img, 0u, uv, img.linear_filter);
  }
  let size = vec2<f32>(f32(img.width), f32(img.height));
  let rho = max(length(dx.xy * size), length(dy.xy * size));
  var lam = log2(max(rho, 1e-30));
  let top = f32(img.levels - 1u);
  lam = select(0.0, clamp(lam, 0.0, top), lam > 0.0);
  if (!img.linear_filter) {
    let k = min(u32(floor(lam + 0.5)), img.levels - 1u);
    return sample_level(img, k, uv, false);
  }
  let k0 = u32(floor(lam));
  let k1 = min(k0 + 1u, img.levels - 1u);
  let a = sample_level(img, k0, uv, true);
  if (k1 == k0) {
    return a;
  }
  let t = lam - f32(k0);
  return a * (1.0 - t) + sample_level(img, k1, uv, true) * t;
}

// -- the texture program interpreter (aether3d/materials/graph.py) -----------------------------

const REGISTER_BIT: u32 = 0x80000000u;
const OP_CONSTANT: u32 = 1u;
const OP_COORDINATES: u32 = 2u;
const OP_CHECKER: u32 = 3u;
const OP_GRADIENT: u32 = 4u;
const OP_STRIPES: u32 = 5u;
const OP_NOISE: u32 = 6u;
const OP_FBM: u32 = 7u;
const OP_VORONOI: u32 = 8u;
const OP_BRICKS: u32 = 9u;
const OP_TILES: u32 = 10u;
const OP_WOOD: u32 = 11u;
const OP_MIX: u32 = 12u;
const OP_RAMP: u32 = 13u;
const OP_MATH: u32 = 14u;
const OP_BUMP: u32 = 15u;
const OP_IMAGE: u32 = 16u;
const OP_SET_COORD: u32 = 32u;
const OP_OUTPUT: u32 = 33u;

var<private> regs: array<vec4<f32>, 16>;
var<private> vm_params: u32;

fn param(k: u32) -> f32 {
  return wordf(vm_params + k);
}

fn param3(k: u32) -> vec3<f32> {
  return vec3<f32>(param(k), param(k + 1u), param(k + 2u));
}

fn luminance(c: vec3<f32>) -> f32 {
  return c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
}

fn operand(w: u32) -> vec4<f32> {
  if ((w & REGISTER_BIT) == 0u) {
    return vec4<f32>(param(w), param(w + 1u), param(w + 2u), param(w + 3u));
  }
  let v = regs[w & 15u];
  let sel = (w >> 8u) & 7u;
  if (sel == 0u) {
    return v;
  }
  var s = 0.0;
  switch (sel) {
    case 1u: { s = v.x; }
    case 2u: { s = v.y; }
    case 3u: { s = v.z; }
    case 4u: { s = v.w; }
    default: { s = luminance(v.xyz); }
  }
  return vec4<f32>(s, s, s, 1.0);
}

fn gradient_value(q: vec3<f32>, kind: u32) -> f32 {
  var v = q.x;
  switch (kind) {
    case 1u: { v = max(q.x, 0.0) * max(q.x, 0.0); }
    case 2u: {
      let c = clamp(q.x, 0.0, 1.0);
      v = c * c * (3.0 - 2.0 * c);
    }
    case 3u: { v = (q.x + q.y) * 0.5; }
    case 4u: { v = atan2(q.y, q.x) / (2.0 * PI) + 0.5; }
    case 5u: { v = max(1.0 - length(q), 0.0); }
    default: {}
  }
  return clamp(v, 0.0, 1.0);
}

fn mix_mode(a: vec4<f32>, b: vec4<f32>, f: f32, mode: u32) -> vec4<f32> {
  let ca = a.xyz;
  let cb = b.xyz;
  var blend = cb;
  switch (mode) {
    case 1u: { blend = ca * cb; }
    case 2u: { blend = ca + cb; }
    case 3u: { blend = ca - cb; }
    case 4u: { blend = 1.0 - (1.0 - ca) * (1.0 - cb); }
    case 5u: { blend = abs(ca - cb); }
    case 6u: { blend = min(ca, cb); }
    case 7u: { blend = max(ca, cb); }
    case 8u: {
      blend = select(1.0 - 2.0 * (1.0 - ca) * (1.0 - cb), 2.0 * ca * cb, ca < vec3<f32>(0.5));
    }
    default: {}
  }
  return vec4<f32>(ca + (blend - ca) * f, a.w + (b.w - a.w) * f);
}

fn math_op(op: u32, a: f32, b: f32, c: f32) -> f32 {
  switch (op) {
    case 0u: { return a + b; }
    case 1u: { return a - b; }
    case 2u: { return a * b; }
    case 3u: { return select(0.0, a / b, b != 0.0); }
    case 4u: {
      let base = max(a, 0.0);
      if (base == 0.0) {
        return select(0.0, 1.0, b == 0.0);
      }
      return min(pow(base, b), 3.0e38);
    }
    case 5u: { return min(a, b); }
    case 6u: { return max(a, b); }
    case 7u: { return abs(a); }
    case 8u: { return 1.0 - a; }
    case 9u: { return min(max(a, b), c); }
    case 10u: { return smooth_step(b, c, a); }
    default: {
      let span = c - b;
      return select(0.0, (a - b) / span, span != 0.0);
    }
  }
}

fn ramp(f: f32, count: u32, mode: u32) -> vec4<f32> {
  let x = clamp(f, 0.0, 1.0);
  var k = 0u;
  for (var i = 0u; i < count; i++) {
    if (param(4u * i) <= x) {
      k = i;
    }
  }
  let value = vec4<f32>(param3(4u * k + 1u), 1.0);
  if (mode == 1u) {
    return value;
  }
  let next = min(k + 1u, count - 1u);
  let span = param(4u * next) - param(4u * k);
  var t = select(0.0, (x - param(4u * k)) / span, span > 0.0);
  if (x < param(0u)) {
    t = 0.0;
  }
  if (mode == 2u) {
    t = t * t * (3.0 - 2.0 * t);
  }
  return value + (vec4<f32>(param3(4u * next + 1u), 1.0) - value) * t;
}

fn bricks(q: vec3<f32>, a: vec4<f32>, b: vec4<f32>, mortar_color: vec4<f32>, seed: u32) -> vec4<f32> {
  // params (after scale/offset): width, height, mortar, smoothness, row offset, variation
  let w = param(6u);
  let h = param(7u);
  let row = floor(q.y / h);
  let shifted = row * param(10u);
  let x = q.x + (shifted - floor(shifted)) * w;
  let col = floor(x / w);
  let lx = x - col * w;
  let ly = q.y - row * h;
  let d = min(min(lx, w - lx), min(ly, h - ly));
  let half_mortar = param(8u) * 0.5;
  let mortar = 1.0 - smooth_step(half_mortar, half_mortar + param(8u) * param(9u), d);
  let tint = cell_random(col, row, seed) * param(11u);
  let brick = a + (b - a) * tint;
  let out = brick + (mortar_color - brick) * mortar;
  return vec4<f32>(out.xyz, mortar);
}

fn wood(q: vec3<f32>, a: vec4<f32>, b: vec4<f32>, axis: u32, seed: u32) -> vec4<f32> {
  var stretch = vec3<f32>(2.0);
  var fine = vec3<f32>(40.0);
  var r2 = 0.0;
  if (axis == 0u) {
    stretch.x = 0.5;
    fine.x = 2.0;
    r2 = q.y * q.y + q.z * q.z;
  } else if (axis == 1u) {
    stretch.y = 0.5;
    fine.y = 2.0;
    r2 = q.x * q.x + q.z * q.z;
  } else {
    stretch.z = 0.5;
    fine.z = 2.0;
    r2 = q.x * q.x + q.y * q.y;
  }
  let warp = fbm(q * stretch, seed, 1u, 3u, 2.0, 0.5);
  let r = sqrt(r2) + param(7u) * (warp - 0.5);
  var ring = r * param(6u);
  ring = ring - floor(ring);
  let grain = noise3(q * fine, seed + 1u, 1u);
  let t = clamp(ring * ring + param(8u) * (grain - 0.5), 0.0, 1.0);
  let out = a + (b - a) * t;
  return vec4<f32>(out.xyz, t);
}

// Run a texture program at coordinate p (with screen derivatives dx, dy).
fn run_program(first: u32, count: u32, p: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> vec4<f32> {
  let programs = header(HEADER_PROGRAMS);
  let params = header(HEADER_PARAMS);
  var coord = p;
  for (var i = 0u; i < count; i++) {
    let at = programs + (first + i) * INSTRUCTION_WORDS;
    let w0 = word(at);
    let op = w0 & 0xffu;
    let aux = w0 >> 8u;
    let dst = word(at + 1u);
    vm_params = params + word(at + 6u);
    let extra = word(at + 7u);
    if (op == OP_SET_COORD) {
      coord = p + param3(0u);
      continue;
    }
    // Operand constants are relative to the parameter region, not to this instruction.
    let saved = vm_params;
    vm_params = params;
    let i0 = operand(word(at + 2u));
    let i1 = operand(word(at + 3u));
    let i2 = operand(word(at + 4u));
    vm_params = saved;
    if (op == OP_OUTPUT) {
      regs[0] = i0;
      continue;
    }
    var result = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    switch (op) {
      case 1u: { result = i0; }
      case 12u: { result = mix_mode(i0, i1, i2.x, aux); }
      case 13u: { result = ramp(i0.x, aux >> 4u, aux & 15u); }
      case 14u: {
        let v = math_op(aux, i0.x, i1.x, i2.x);
        result = vec4<f32>(v, v, v, 1.0);
      }
      case 15u: {
        let depth = param(0u);
        let step = param(1u);
        let sx = -depth * (i1.x - i0.x) / step;
        let sy = -depth * (i2.x - i0.x) / step;
        let inv = 1.0 / sqrt(sx * sx + sy * sy + 1.0);
        result = vec4<f32>(sx * inv * 0.5 + 0.5, sy * inv * 0.5 + 0.5, inv * 0.5 + 0.5, 1.0);
      }
      default: {
        // Pattern nodes: the coordinate through the node's scale and offset.
        let scale = param3(0u);
        let q = coord * scale + param3(3u);
        switch (op) {
          case 2u: { result = vec4<f32>(q, 1.0); }
          case 3u: {
            var c = floor(q.x) + floor(q.y);
            if (aux == 3u) {
              c = c + floor(q.z);
            }
            let mask = c - 2.0 * floor(c * 0.5);
            result = vec4<f32>((i0 + (i1 - i0) * mask).xyz, mask);
          }
          case 4u: {
            let v = gradient_value(q, aux);
            result = vec4<f32>(v, v, v, 1.0);
          }
          case 5u: {
            var a = q.x;
            if (aux == 1u) {
              a = q.y;
            } else if (aux == 2u) {
              a = q.z;
            } else if (aux == 3u) {
              a = q.x + q.y;
            }
            let mask = select(0.0, 1.0, (a - floor(a)) >= param(6u));
            result = vec4<f32>((i0 + (i1 - i0) * mask).xyz, mask);
          }
          case 6u: {
            let v = noise3(q, extra, aux & 15u);
            result = vec4<f32>(v, v, v, 1.0);
          }
          case 7u: {
            let v = fbm(q, extra, aux & 15u, aux >> 4u, param(6u), param(7u));
            result = vec4<f32>(v, v, v, 1.0);
          }
          case 8u: { result = vec4<f32>(voronoi(q, extra, param(6u), aux), 1.0); }
          case 9u, 10u: { result = bricks(q, i0, i1, i2, extra); }
          case 11u: { result = wood(q, i0, i1, aux, extra); }
          case 16u: { result = sample_image(extra, q, dx * scale, dy * scale); }
          default: {}
        }
      }
    }
    regs[dst] = result;
  }
  return regs[0];
}

// -- materials: coordinates, bindings and surface values (aether3d/rendering/surface.py) -------

struct SurfaceInput {
  instance: u32,
  bary: vec3<f32>,
  dbdx: vec3<f32>,
  dbdy: vec3<f32>,
  p0: vec3<f32>,      // camera-relative world corners
  p1: vec3<f32>,
  p2: vec3<f32>,
  uv0: vec2<f32>,
  uv1: vec2<f32>,
  uv2: vec2<f32>,
  normal: vec3<f32>,  // outward unit shading normal (world)
  vnormal: vec3<f32>,  // the interpolated outward normal, not renormalized (MikkTSpace N)
  tangent: vec4<f32>,  // the interpolated MikkTSpace tangent and its sign (zero: none)
  color: vec4<f32>,    // the interpolated vertex colour (white without vertex colours)
};

struct Corners {
  c0: vec3<f32>,
  c1: vec3<f32>,
  c2: vec3<f32>,
};

struct Coords {
  p: vec3<f32>,
  dx: vec3<f32>,
  dy: vec3<f32>,
};

fn object_point(inst: Instance, p: vec3<f32>) -> vec3<f32> {
  // M^-1 (p - t) with M^-1 = transpose(normal matrix).
  let d = p - vec3<f32>(inst.model[0].w, inst.model[1].w, inst.model[2].w);
  return d.x * inst.normal[0].xyz + d.y * inst.normal[1].xyz + d.z * inst.normal[2].xyz;
}

fn object_direction(inst: Instance, n: vec3<f32>) -> vec3<f32> {
  // M^T n: a world normal in object space (normalized).
  return safe_normalize(n.x * inst.model[0].xyz + n.y * inst.model[1].xyz + n.z * inst.model[2].xyz);
}

fn binding_transform(b: u32, c: vec3<f32>) -> vec3<f32> {
  let m = b + 4u;
  return vec3<f32>(
    dot(word3(m), c) + wordf(m + 3u),
    dot(word3(m + 4u), c) + wordf(m + 7u),
    dot(word3(m + 8u), c) + wordf(m + 11u)
  );
}

// The points a binding's coordinates come from (uv, object or world space).
fn source_corners(coordinates: u32, s: SurfaceInput) -> Corners {
  if (coordinates == COORD_UV) {
    return Corners(vec3<f32>(s.uv0, 0.0), vec3<f32>(s.uv1, 0.0), vec3<f32>(s.uv2, 0.0));
  }
  if (coordinates == COORD_OBJECT) {
    let inst = instances[s.instance];
    return Corners(object_point(inst, s.p0), object_point(inst, s.p1), object_point(inst, s.p2));
  }
  let eye = camera_eye();
  return Corners(s.p0 + eye, s.p1 + eye, s.p2 + eye);
}

// Corner coordinates of one projection (axis 0-2 for triplanar), transformed by the binding.
fn projected_corners(b: u32, projection: u32, axis: u32, src: Corners, sign: vec3<f32>) -> Corners {
  var a = src;
  if (projection == PROJ_TRIPLANAR) {
    if (axis == 0u) {
      a = Corners(vec3<f32>(sign.x * src.c0.y, src.c0.z, 0.0), vec3<f32>(sign.x * src.c1.y, src.c1.z, 0.0), vec3<f32>(sign.x * src.c2.y, src.c2.z, 0.0));
    } else if (axis == 1u) {
      a = Corners(vec3<f32>(-sign.y * src.c0.x, src.c0.z, 0.0), vec3<f32>(-sign.y * src.c1.x, src.c1.z, 0.0), vec3<f32>(-sign.y * src.c2.x, src.c2.z, 0.0));
    } else {
      a = Corners(vec3<f32>(sign.z * src.c0.x, src.c0.y, 0.0), vec3<f32>(sign.z * src.c1.x, src.c1.y, 0.0), vec3<f32>(sign.z * src.c2.x, src.c2.y, 0.0));
    }
  } else if (projection == PROJ_XY) {
    a = Corners(vec3<f32>(src.c0.xy, 0.0), vec3<f32>(src.c1.xy, 0.0), vec3<f32>(src.c2.xy, 0.0));
  } else if (projection == PROJ_XZ) {
    a = Corners(vec3<f32>(src.c0.xz, 0.0), vec3<f32>(src.c1.xz, 0.0), vec3<f32>(src.c2.xz, 0.0));
  } else if (projection == PROJ_YZ) {
    a = Corners(vec3<f32>(src.c0.yz, 0.0), vec3<f32>(src.c1.yz, 0.0), vec3<f32>(src.c2.yz, 0.0));
  }
  return Corners(binding_transform(b, a.c0), binding_transform(b, a.c1), binding_transform(b, a.c2));
}

fn interpolate(c: Corners, s: SurfaceInput) -> Coords {
  return Coords(
    s.bary.x * c.c0 + s.bary.y * c.c1 + s.bary.z * c.c2,
    s.dbdx.x * c.c0 + s.dbdx.y * c.c1 + s.dbdx.z * c.c2,
    s.dbdy.x * c.c0 + s.dbdy.y * c.c1 + s.dbdy.z * c.c2
  );
}

struct Projections {
  count: u32,             // 1, or 3 for triplanar
  weights: vec3<f32>,
  sign: vec3<f32>,
  src: Corners,
  projection: u32,
};

fn projections(b: u32, s: SurfaceInput) -> Projections {
  let flags = word(b);
  let coordinates = flags & 3u;
  var projection = (flags >> 4u) & 7u;
  if (coordinates == COORD_UV) {
    projection = PROJ_VOLUME;  // uv corners are used as they are
  }
  let src = source_corners(coordinates, s);
  if (projection != PROJ_TRIPLANAR) {
    return Projections(1u, vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0), src, projection);
  }
  var n = s.normal;
  if (coordinates == COORD_OBJECT) {
    n = object_direction(instances[s.instance], s.normal);
  }
  let g = cross(src.c1 - src.c0, src.c2 - src.c0);
  let sign = select(vec3<f32>(-1.0), vec3<f32>(1.0), g >= vec3<f32>(0.0));
  var w = pow(abs(n), vec3<f32>(wordf(b + 3u)));
  w = w / max(w.x + w.y + w.z, 1e-30);
  w = select(vec3<f32>(0.0), w, w > vec3<f32>(TRIPLANAR_EPSILON));
  return Projections(3u, w, sign, src, projection);
}

fn channel(v: vec4<f32>, which: u32) -> f32 {
  switch (which) {
    case 1u: { return v.x; }
    case 2u: { return v.y; }
    case 3u: { return v.z; }
    case 4u: { return v.w; }
    default: { return luminance(v.xyz); }
  }
}

struct Frame3 {
  t: vec3<f32>,
  b: vec3<f32>,
  valid: bool,
};

// Tangent frame from the triangle's texture-coordinate gradients, orthonormalized against n.
fn tangent_frame(c: Corners, s: SurfaceInput, n: vec3<f32>) -> Frame3 {
  let du1 = c.c1.xy - c.c0.xy;
  let du2 = c.c2.xy - c.c0.xy;
  let det = du1.x * du2.y - du2.x * du1.y;
  var valid = abs(det) > 1e-20;
  let inv = select(0.0, 1.0 / det, valid);
  let e1 = s.p1 - s.p0;
  let e2 = s.p2 - s.p0;
  var t = (e1 * du2.y - e2 * du1.y) * inv;
  let b = (e2 * du1.x - e1 * du2.x) * inv;
  t = t - n * dot(t, n);
  let t_len = length(t);
  valid = valid && t_len > 1e-20;
  t = select(vec3<f32>(0.0), t / t_len, t_len > 1e-20);
  let bitangent = cross(n, t);
  let handed = select(1.0, -1.0, dot(bitangent, b) < 0.0);
  return Frame3(t, bitangent * handed, valid);
}

// Whether a material's normal map is read in the MikkTSpace basis (a UV binding).
fn needs_tangents(m: u32) -> bool {
  let sb = slot_binding(m, SLOT_NORMAL);
  if (sb == NO_BINDING) {
    return false;
  }
  return (word(header(HEADER_BINDINGS) + sb * BINDING_WORDS) & BINDING_MIKK) != 0u;
}

// The MikkTSpace pixel-shader transform (rendering/tangents.py): interpolated T and N,
// B = sign * cross(N, T), n = normalize(x T + y B + z N).
fn mikk_normal(rgba: vec4<f32>, s: SurfaceInput, strength: f32) -> vec3<f32> {
  var tn = rgba.xyz * 2.0 - 1.0;
  tn = vec3<f32>(tn.xy * strength, tn.z);
  let t = s.tangent.xyz;
  let n = s.vnormal;
  let b = select(1.0, -1.0, s.tangent.w < 0.0) * cross(n, t);
  let mapped = t * tn.x + b * tn.y + n * tn.z;
  let len = length(mapped);
  if (length(t) > 1e-12 && len > 1e-12) {
    return mapped / len;
  }
  return s.normal;
}

fn mapped_normal(rgba: vec4<f32>, frame: Frame3, n: vec3<f32>, strength: f32) -> vec3<f32> {
  var tn = rgba.xyz * 2.0 - 1.0;
  tn = vec3<f32>(tn.xy * strength, tn.z);
  let mapped = frame.t * tn.x + frame.b * tn.y + n * tn.z;
  let len = length(mapped);
  if (frame.valid && len > 1e-12) {
    return mapped / len;
  }
  return n;
}

struct SurfaceValues {
  base: vec3<f32>,
  alpha: f32,
  metallic: f32,
  roughness: f32,
  occlusion: f32,
  emissive: vec3<f32>,
  normal: vec3<f32>,  // outward
};

fn slot_binding(m: u32, slot: u32) -> u32 {
  return word(material_base(m) + 16u + slot);
}

// A binding applies unless it maps by UV on a mesh without UVs (reported on the CPU).
fn binding_usable(slot_binding: u32, has_uvs: bool) -> bool {
  if (slot_binding == NO_BINDING) {
    return false;
  }
  let b = header(HEADER_BINDINGS) + slot_binding * BINDING_WORDS;
  return has_uvs || (word(b) & 3u) != COORD_UV;
}

fn material_flags(m: u32) -> u32 {
  return word(material_base(m));
}

// Every texture of a material is evaluated by the one loop below, so the texture-program
// interpreter (run_program, which holds every node type) appears once in the compiled shader.
// Shader compilers inline every call, and D3D12's FXC also unrolls loops whose trip count it
// knows: one call per slot and projection made the shader dozens of times larger, and took
// minutes to compile there. The loop visits the bound slots through a mask of loaded data (in
// slot order; each slot sets its own values), and each slot's projections.
// What a material looks like at a surface point: factors x textures (surface_textures),
// times the vertex colour (glTF COLOR_0 multiplies base colour and opacity).
fn evaluate_surface(m: u32, s: SurfaceInput, has_uvs: bool, alpha_only: bool) -> SurfaceValues {
  var v = surface_textures(m, s, has_uvs, alpha_only);
  v.base = v.base * s.color.rgb;
  v.alpha = v.alpha * s.color.a;
  return v;
}

fn surface_textures(m: u32, s: SurfaceInput, has_uvs: bool, alpha_only: bool) -> SurfaceValues {
  let mb = material_base(m);
  var v = SurfaceValues(
    word3(mb + 1u), wordf(mb + 4u), wordf(mb + 6u), wordf(mb + 7u), 1.0, word3(mb + 10u), s.normal
  );
  if ((word(mb) & MATERIAL_TEXTURED) == 0u) {
    return v;
  }
  var pending = 0u;
  for (var slot = 0u; slot < 7u; slot++) {
    if (binding_usable(slot_binding(m, slot), has_uvs)) {
      pending = pending | (1u << slot);
    }
  }
  if (alpha_only) {
    pending = pending & ((1u << SLOT_BASE_COLOR) | (1u << SLOT_OPACITY));
  }
  while (pending != 0u) {
    let slot = firstTrailingBit(pending);
    pending = pending & (pending - 1u);
    let sb = slot_binding(m, slot);
    let b = header(HEADER_BINDINGS) + sb * BINDING_WORDS;
    let first = word(b + 1u);
    let count = word(b + 2u);
    let pr = projections(b, s);  // one projection (weight 1), or three triplanar ones
    var rgba = vec4<f32>(0.0);
    var normal = vec3<f32>(0.0);
    for (var axis = 0u; axis < pr.count; axis++) {
      let w = pr.weights[axis];
      if (w > 0.0) {
        let corners = projected_corners(b, pr.projection, axis, pr.src, pr.sign);
        let c = interpolate(corners, s);
        let value = run_program(first, count, c.p, c.dx, c.dy);
        if (slot == SLOT_NORMAL) {
          if ((word(b) & BINDING_MIKK) != 0u) {
            normal = normal + mikk_normal(value, s, wordf(mb + 13u)) * w;
          } else {
            let frame = tangent_frame(corners, s, s.normal);
            normal = normal + mapped_normal(value, frame, s.normal, wordf(mb + 13u)) * w;
          }
        } else {
          rgba = rgba + value * w;
        }
      }
    }
    switch (slot) {
      case 0u: {  // SLOT_BASE_COLOR: an image's alpha is coverage
        v.base = v.base * rgba.xyz;
        if ((word(b) & BINDING_IMAGE) != 0u) {
          v.alpha = v.alpha * rgba.w;
        }
      }
      case 1u: {  // SLOT_OPACITY
        v.alpha = v.alpha * channel(rgba, binding_channel(sb));
      }
      case 2u: {  // SLOT_METALLIC
        v.metallic = v.metallic * channel(rgba, binding_channel(sb));
      }
      case 3u: {  // SLOT_ROUGHNESS
        v.roughness = v.roughness * channel(rgba, binding_channel(sb));
      }
      case 4u: {  // SLOT_NORMAL: the blended tilted normals, or the surface normal
        let len = length(normal);
        if (len > 1e-12) {
          v.normal = normal / len;
        }
      }
      case 5u: {  // SLOT_EMISSIVE
        v.emissive = v.emissive * rgba.xyz;
      }
      default: {  // SLOT_OCCLUSION
        v.occlusion = 1.0 + wordf(mb + 14u) * (channel(rgba, binding_channel(sb)) - 1.0);
      }
    }
  }
  return v;
}

fn binding_channel(slot_binding: u32) -> u32 {
  return (word(header(HEADER_BINDINGS) + slot_binding * BINDING_WORDS) >> 8u) & 7u;
}
