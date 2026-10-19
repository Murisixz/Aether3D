// Aether3D GPU renderer -- the scene pass (appended to common.wgsl and material.wgsl).
//
// Vertex stage: camera-relative positions, world normals, the exact face
// normal of every triangle, its corner UVs and its material (build_vertex).
// Fragment stage: the material (textures, normal maps, alpha cut-outs,
// single-sided culling; material.wgsl), then one of two shading models in
// linear light (docs/RENDERING.md, docs/MATERIALS.md):
//
//   simple:   base * (ambient * ao + sum(light * max(n.l, 0) * attenuation * shadow))
//             + sum(light * specular * max(n.h, 0)^shininess * attenuation * shadow)
//   standard: glTF metallic-roughness (GGX, height-correlated Smith, Schlick), times pi
//
// plus emission, with two-sided lighting (normals turned toward the camera).
// Outputs are the linear radiance and the object index + 1 of every sample
// (0 = background).

const MIN_LIGHT_DISTANCE: f32 = 1e-4;

struct Frame {
  view_proj: mat4x4<f32>,  // camera-relative world -> clip; depth is reversed (near 1, far 0)
  forward: vec4<f32>,      // xyz: camera forward; w: 1 perspective, 0 orthographic
  ambient: vec4<f32>,      // rgb: linear ambient light (color x intensity)
  light_count: u32,
  pcf_radius: i32,         // shadow filter radius in texels
  shadow_texel: f32,       // one shadow-map texel in texture coordinates
  _pad: u32,
  eye: vec4<f32>,          // xyz: the camera position in world space (for world coordinates)
  atmo: Atmo,              // scene.world.atmosphere (atmosphere.wgsl)
};

struct Light {
  radiance: vec3<f32>,     // linear color x intensity
  kind: u32,               // 0 directional, 1 point, 2 spot, 3 area
  vector: vec3<f32>,       // directional: direction of travel; others: camera-relative position
  range: f32,              // point, spot: distance where the light fades out, 0 = no range
  falloff: u32,            // point, spot: 1 inverse square, 0 none
  shadow: i32,             // first shadow-map layer, -1 = casts no shadow
  cascades: u32,           // consecutive layers of this light (cascades by view depth)
  shape: u32,              // area: 0 square or rectangle, 1 disk or ellipse
  axis: vec3<f32>,         // spot: direction of travel; area: its normal (it emits that way)
  cos_outer: f32,          // spot: cosine of the cone's half angle
  axis_x: vec3<f32>,       // area: half extent along the light's local X
  spot_smooth: f32,        // spot: the smoothstep's width in cos (0: a hard edge)
  axis_y: vec3<f32>,       // area: half extent along its local Z
  area: f32,               // area: emitting area (m^2)
  spread: vec4<f32>,       // area: tan(spread / 2) (0: 180 degrees), its normalization
};

const AREA_POLYGON: u32 = 16u;  // corners of a disk or ellipse (surface.py)
const AREA_SCALE: f32 = 1.0130016;  // their radius, for the ellipse's area (surface.AREA_SCALE)
const SPREAD_QUADRATURE: u32 = 12u;  // grid points per side for spread lights (surface.py)

struct ShadowMap {
  matrix: mat4x4<f32>,     // camera-relative world -> shadow clip (x, y in [-1, 1], depth in [0, 1])
  bias: f32,               // constant depth bias, shadow depth units
  normal_offset: f32,      // world units, scaled by the sine of the light angle
  split: f32,              // cascades: the far view depth this layer covers
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(2) var<storage, read> lights: array<Light>;
@group(0) @binding(3) var<storage, read> shadow_maps: array<ShadowMap>;
@group(0) @binding(4) var shadow_texture: texture_depth_2d_array;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;

fn camera_eye() -> vec3<f32> {
  return frame.eye.xyz;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOut {
  return build_vertex(vertex, instance, frame.view_proj);
}

// The point on the smooth surface that the vertex normals describe, above a flat triangle
// (Hanika, "Hacking the Shadow Terminator", Ray Tracing Gems II, 2021): the fragment is moved
// out of each corner's tangent plane it lies below, weighted by its barycentrics. Flat-shaded
// triangles (vertex normals = face normal) are left unchanged. Shadow lookups start there, so
// a coarse, smoothly shaded mesh does not shadow itself along its facets near the terminator.
// `side` (+1 or -1) turns the normals toward the visible, lit side of the surface.
fn smooth_surface_point(in: VertexOut, side: f32) -> vec3<f32> {
  let p = in.position;
  let b = in.barycentric;
  let n0 = in.n0 * side;
  let n1 = in.n1 * side;
  let n2 = in.n2 * side;
  let d0 = min(0.0, dot(p - in.p0, n0));
  let d1 = min(0.0, dot(p - in.p1, n1));
  let d2 = min(0.0, dot(p - in.p2, n2));
  return p - (b.x * d0) * n0 - (b.y * d1) * n1 - (b.z * d2) * n2;
}

// The layer of a light's shadow for a point at camera-relative `position`: with cascades,
// the first whose far split lies beyond the point's view depth.
fn shadow_layer(light: Light, position: vec3<f32>) -> i32 {
  if (light.cascades <= 1u) {
    return light.shadow;
  }
  let depth = dot(position, frame.forward.xyz);
  for (var c = 0u; c < light.cascades; c++) {
    let layer = light.shadow + i32(c);
    if (depth <= shadow_maps[layer].split) {
      return layer;
    }
  }
  return -1;  // beyond the shadowed distance
}

// Fraction of a directional light reaching `position` (1 = fully lit), from its shadow map.
// `n` is the shading normal, on the light's side of the surface (n . l > 0).
fn shadow_visibility(layer: i32, position: vec3<f32>, n: vec3<f32>, l: vec3<f32>) -> f32 {
  let shadow = shadow_maps[layer];
  let cos_angle = clamp(dot(n, l), 0.0, 1.0);
  let sin_angle = sqrt(max(1.0 - cos_angle * cos_angle, 0.0));
  let offset_position = position + n * (shadow.normal_offset * sin_angle);
  let p = shadow.matrix * vec4<f32>(offset_position, 1.0);
  let uv = vec2<f32>(0.5 + 0.5 * p.x, 0.5 - 0.5 * p.y);
  let depth = p.z - shadow.bias;
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || depth > 1.0) {
    return 1.0;  // outside the fitted map: nothing in the scene can shadow it
  }
  let r = frame.pcf_radius;
  var lit = 0.0;
  for (var y = -r; y <= r; y++) {
    for (var x = -r; x <= r; x++) {
      let tap = uv + vec2<f32>(f32(x), f32(y)) * frame.shadow_texel;
      lit += textureSampleCompareLevel(shadow_texture, shadow_sampler, tap, layer, depth);
    }
  }
  return lit / f32((2 * r + 1) * (2 * r + 1));
}

fn view_direction(position: vec3<f32>) -> vec3<f32> {
  if (frame.forward.w > 0.5) {
    return safe_normalize(-position);
  }
  return -frame.forward.xyz;
}

struct LightSample {
  l: vec3<f32>,        // unit direction toward the light
  attenuation: f32,
};

fn light_at(light: Light, position: vec3<f32>) -> LightSample {
  if (light.kind == 0u) {
    return LightSample(-light.vector, 1.0);
  }
  let to_light = light.vector - position;
  let distance = length(to_light);
  let l = to_light / max(distance, 1e-30);
  let d = max(distance, MIN_LIGHT_DISTANCE);
  var attenuation = 1.0;
  if (light.falloff == 1u) {
    attenuation = 1.0 / (d * d);
  }
  if (light.range > 0.0) {
    let ratio = d / light.range;
    let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    attenuation = attenuation * window * window;
  }
  if (light.kind == 2u) {
    attenuation = attenuation * spot_mask(light, -dot(l, light.axis));
  }
  return LightSample(l, attenuation);
}

// A spot's mask at the cosine between its axis and the direction from it (surface.py).
fn spot_mask(light: Light, cos_t: f32) -> f32 {
  if (cos_t <= light.cos_outer) {
    return 0.0;
  }
  if (light.spot_smooth > 0.0) {
    let x = clamp((cos_t - light.cos_outer) / light.spot_smooth, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
  }
  return 1.0;
}

// Corner k of an area light's polygon, relative to `position` (surface.area_corners).
fn area_corner(light: Light, k: u32, count: u32, position: vec3<f32>) -> vec3<f32> {
  let c = light.vector - position;
  if (count == 4u) {
    let sx = select(-1.0, 1.0, k == 1u || k == 2u);
    let sy = select(-1.0, 1.0, k >= 2u);
    return c + sx * light.axis_x + sy * light.axis_y;
  }
  let t = f32(k) * (2.0 * PI / f32(AREA_POLYGON));
  let scale = AREA_SCALE;
  return c + (scale * cos(t)) * light.axis_x + (scale * sin(t)) * light.axis_y;
}

// theta (n . gamma) of the edge a -> b (Lambert's polygon formula).
fn area_edge(a: vec3<f32>, b: vec3<f32>, n: vec3<f32>) -> f32 {
  let g = cross(a, b);
  let len = length(g);
  if (len <= 0.0) {
    return 0.0;
  }
  return atan2(len, dot(a, b)) * dot(g, n) / len;
}

// A spread area light's radiance factor toward cos_a from its normal (scene_data.py).
fn spread_factor(light: Light, cos_a: f32) -> f32 {
  if (!(cos_a > 0.0)) {
    return 0.0;
  }
  if (light.spread.x <= 0.0) {
    return 1.0;
  }
  let tan_a = sqrt(max(1.0 - cos_a * cos_a, 0.0)) / max(cos_a, 1e-30);
  return max(1.0 - tan_a / light.spread.x, 0.0) * light.spread.y;
}

struct SpreadWindow {
  lo: vec2<f32>,  // (u0, w0): the part of the light that reaches the point (half extents)
  hi: vec2<f32>,  // (u1, w1)
  area: f32,
  ok: bool,
};

fn spread_window(light: Light, position: vec3<f32>) -> SpreadWindow {
  let rel = position - light.vector;
  let t = dot(rel, light.axis);
  let lx = length(light.axis_x);
  let ly = length(light.axis_y);
  let qu = dot(rel, light.axis_x) / (lx * lx);
  let qw = dot(rel, light.axis_y) / (ly * ly);
  let reach = max(t, 0.0) * light.spread.x;
  let lo = vec2<f32>(max(qu - reach / lx, -1.0), max(qw - reach / ly, -1.0));
  let hi = vec2<f32>(min(qu + reach / lx, 1.0), min(qw + reach / ly, 1.0));
  let ok = t > 0.0 && hi.x > lo.x && hi.y > lo.y;
  var area = 0.0;
  if (ok) {
    area = (hi.x - lo.x) * lx * (hi.y - lo.y) * ly;
  }
  return SpreadWindow(lo, hi, area, ok);
}

// G of a spread area light, its radiance factor included: the midpoint rule over the part
// of it that reaches the point (surface.spread_irradiance).
fn spread_irradiance(light: Light, position: vec3<f32>, n: vec3<f32>) -> f32 {
  let win = spread_window(light, position);
  if (!win.ok) {
    return 0.0;
  }
  let k = SPREAD_QUADRATURE;
  let du = (win.hi.x - win.lo.x) / f32(k);
  let dw = (win.hi.y - win.lo.y) / f32(k);
  let c = light.vector - position;
  var total = 0.0;
  for (var i = 0u; i < k; i++) {
    let u = win.lo.x + (f32(i) + 0.5) * du;
    for (var j = 0u; j < k; j++) {
      let w = win.lo.y + (f32(j) + 0.5) * dw;
      let v = c + u * light.axis_x + w * light.axis_y;
      let d2 = max(dot(v, v), MIN_LIGHT_DISTANCE * MIN_LIGHT_DISTANCE);
      let d = sqrt(d2);
      let cos_p = max(dot(v, n) / d, 0.0);
      let cos_l = -dot(v, light.axis) / d;
      var term = spread_factor(light, cos_l) * max(cos_l, 0.0) * cos_p / d2;
      if (light.shape == 1u) {
        let r = sqrt(u * u + w * w);
        let rs = max(r, 1e-30);
        let across = abs(u / rs) * du + abs(w / rs) * dw;
        term = term * clamp(0.5 - (r - 1.0) / max(across, 1e-30), 0.0, 1.0);
      }
      total += term;
    }
  }
  return total * (win.area / f32(k * k));
}

// Irradiance of an area light of unit radiance, unshadowed: the polygon clipped by the
// tangent plane (surface.area_irradiance). Zero behind the one-sided light.
fn area_irradiance(light: Light, position: vec3<f32>, n: vec3<f32>) -> f32 {
  if (dot(position - light.vector, light.axis) <= 0.0) {
    return 0.0;
  }
  if (light.spread.x > 0.0) {
    return spread_irradiance(light, position, n);
  }
  let count = select(4u, AREA_POLYGON, light.shape == 1u);
  var total = 0.0;
  var exit_x = vec3<f32>(0.0);
  var entry_x = vec3<f32>(0.0);
  var clipped = false;
  for (var i = 0u; i < count; i++) {
    let a = area_corner(light, i, count, position);
    let b = area_corner(light, (i + 1u) % count, count, position);
    let da = dot(a, n);
    let db = dot(b, n);
    let above_a = da > 0.0;
    let above_b = db > 0.0;
    var f = 0.0;
    if (above_a != above_b && da != db) {
      f = da / (da - db);
    }
    let x = a + f * (b - a);
    if (above_a || above_b) {
      total += area_edge(select(x, a, above_a), select(x, b, above_b), n);
    }
    if (above_a && !above_b) {
      exit_x = x;
      clipped = true;
    }
    if (!above_a && above_b) {
      entry_x = x;
    }
  }
  if (clipped) {
    total += area_edge(exit_x, entry_x, n);
  }
  return 0.5 * abs(total);
}

struct AreaHighlight {
  l: vec3<f32>,      // unit direction to the representative point
  weight: f32,       // cos at the light over the squared distance
  distance: f32,
};

// The representative point of an area light for glossy highlights (surface.area_highlight).
fn area_highlight(light: Light, position: vec3<f32>, n: vec3<f32>, v: vec3<f32>) -> AreaHighlight {
  let r = 2.0 * dot(n, v) * n - v;
  let o = light.vector - position;  // point -> centre
  let denom = dot(r, light.axis);
  var den = denom;
  if (abs(den) <= 1e-12) {
    den = 1e-12;
  }
  var t = dot(o, light.axis) / den;
  if (!(denom < -1e-12 && t > 0.0)) {
    t = 0.0;
  }
  let q = r * t - o;
  var u = dot(q, light.axis_x) / dot(light.axis_x, light.axis_x);
  var w = dot(q, light.axis_y) / dot(light.axis_y, light.axis_y);
  var win = SpreadWindow(vec2<f32>(-1.0), vec2<f32>(1.0), 0.0, true);
  if (light.spread.x > 0.0) {
    win = spread_window(light, position);
    u = clamp(u, win.lo.x, win.hi.x);
    w = clamp(w, win.lo.y, win.hi.y);
  }
  if (light.shape == 1u) {
    let rr = sqrt(u * u + w * w);
    if (rr > 1.0) {
      u = u / rr;
      w = w / rr;
    }
  } else {
    u = clamp(u, -1.0, 1.0);
    w = clamp(w, -1.0, 1.0);
  }
  let lv = o + u * light.axis_x + w * light.axis_y;
  let d2 = max(dot(lv, lv), MIN_LIGHT_DISTANCE * MIN_LIGHT_DISTANCE);
  let d = sqrt(d2);
  let l = lv / d;
  let cos_l = max(-dot(l, light.axis), 0.0);
  if (light.spread.x > 0.0) {
    return AreaHighlight(l, select(0.0, spread_factor(light, cos_l) * cos_l / d2, win.ok), d);
  }
  return AreaHighlight(l, cos_l / d2, d);
}

// Light linking: whether light i leaves this instance unlit (Instance.normal[0..1].w).
fn unlinked(links: vec2<u32>, i: u32) -> bool {
  if (i >= 64u) {
    return false;
  }
  let word = select(links.y, links.x, i < 32u);
  return ((word >> (i % 32u)) & 1u) != 0u;
}

// The simple model (Aether3D 0.5): Lambert plus an optional Blinn-Phong highlight.
fn shade_simple(
  base: vec3<f32>, specular: f32, shininess: f32, occlusion: f32,
  position: vec3<f32>, n: vec3<f32>, shadow_origin: vec3<f32>, links: vec2<u32>
) -> vec3<f32> {
  let v = view_direction(position);
  var diffuse = atmo_ambient(frame.atmo, frame.ambient.rgb, n) * occlusion;
  var highlight = vec3<f32>(0.0);
  for (var i = 0u; i < frame.light_count; i++) {
    let light = lights[i];
    if (unlinked(links, i)) {
      continue;
    }
    if (light.kind == 3u) {  // an area light: the polygon's irradiance, a highlight
      diffuse += light.radiance * (area_irradiance(light, position, n) / light.area);
      if (specular > 0.0) {
        let ah = area_highlight(light, position, n, v);
        if (dot(n, ah.l) > 0.0) {
          let h = ah.l + v;
          let ndh = dot(n, h) / max(length(h), 1e-30);
          if (ndh > 0.0) {
            highlight += light.radiance * (specular * pow(ndh, shininess) * ah.weight);
          }
        }
      }
      continue;
    }
    let ls = light_at(light, position);
    let ndl = dot(n, ls.l);
    if (ndl <= 0.0) {
      continue;
    }
    var visibility = 1.0;
    if (light.shadow >= 0) {
      let layer = shadow_layer(light, position);
      if (layer >= 0) {
        visibility = shadow_visibility(layer, shadow_origin, n, ls.l);
      }
    }
    let weight = ls.attenuation * visibility;
    diffuse += light.radiance * (ndl * weight);
    if (specular > 0.0) {
      let h = ls.l + v;
      let ndh = dot(n, h) / max(length(h), 1e-30);
      if (ndh > 0.0) {
        highlight += light.radiance * (specular * pow(ndh, shininess) * weight);
      }
    }
  }
  return base * diffuse + highlight;
}

// Karis' analytic fit of the split-sum environment BRDF: F0 * x + y.
fn env_brdf(roughness: f32, ndv: f32) -> vec2<f32> {
  let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022) + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
  return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// The standard model: glTF 2.0 metallic-roughness, times pi (see docs/MATERIALS.md).
fn shade_standard(
  sv: SurfaceValues, specular: f32, position: vec3<f32>, n: vec3<f32>, shadow_origin: vec3<f32>,
  links: vec2<u32>
) -> vec3<f32> {
  let v = view_direction(position);
  let metallic = clamp(sv.metallic, 0.0, 1.0);
  let rough = clamp(sv.roughness, MIN_ROUGHNESS, 1.0);
  let alpha = rough * rough;
  let a2 = alpha * alpha;
  let c_diff = sv.base * (1.0 - metallic);
  let f0 = vec3<f32>(0.08 * specular * (1.0 - metallic)) + sv.base * metallic;
  let ndv = max(dot(n, v), 1e-4);
  let ab = env_brdf(rough, ndv);
  var out = atmo_ambient(frame.atmo, frame.ambient.rgb, n) * sv.occlusion * (c_diff + f0 * ab.x + ab.y);
  for (var i = 0u; i < frame.light_count; i++) {
    let light = lights[i];
    if (unlinked(links, i)) {
      continue;
    }
    if (light.kind == 3u) {
      out += area_standard(light, position, n, v, c_diff, f0, alpha, a2, ndv);
      continue;
    }
    let ls = light_at(light, position);
    let ndl = dot(n, ls.l);
    if (ndl <= 0.0) {
      continue;
    }
    var visibility = 1.0;
    if (light.shadow >= 0) {
      let layer = shadow_layer(light, position);
      if (layer >= 0) {
        visibility = shadow_visibility(layer, shadow_origin, n, ls.l);
      }
    }
    let h = safe_normalize(ls.l + v);
    let ndh = max(dot(n, h), 0.0);
    let vdh = max(dot(v, h), 0.0);
    let x = 1.0 - vdh;
    let x2 = x * x;
    let fresnel = f0 + (1.0 - f0) * (x2 * x2 * x);
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    let distribution = a2 / (PI * d * d);
    let vis = 0.5 / max(
      ndl * sqrt(ndv * ndv * (1.0 - a2) + a2) + ndv * sqrt(ndl * ndl * (1.0 - a2) + a2), 1e-30
    );
    let brdf = (1.0 - fresnel) * c_diff + PI * fresnel * distribution * vis;
    out += light.radiance * (ndl * ls.attenuation * visibility) * brdf;
  }
  return out;
}

// An area light on the metallic-roughness model (surface._area_standard).
fn area_standard(
  light: Light, position: vec3<f32>, n: vec3<f32>, v: vec3<f32>, c_diff: vec3<f32>,
  f0: vec3<f32>, alpha: f32, a2: f32, ndv: f32
) -> vec3<f32> {
  let g = area_irradiance(light, position, n) / light.area;
  let ah = area_highlight(light, position, n, v);
  let h = safe_normalize(ah.l + v);
  let ndl = max(dot(n, ah.l), 0.0);
  let ndh = max(dot(n, h), 0.0);
  let vdh = max(dot(v, h), 0.0);
  let x = 1.0 - vdh;
  let x2 = x * x;
  let fresnel = f0 + (1.0 - f0) * (x2 * x2 * x);
  let wide = clamp(alpha + sqrt(light.area / PI) / (2.0 * ah.distance), 0.0, 1.0);
  let a2w = wide * wide;
  let dd = ndh * ndh * (a2w - 1.0) + 1.0;
  let distribution = a2w / (PI * dd * dd) * (alpha * alpha) / max(a2w, 1e-30);
  let vis = 0.5 / max(
    ndl * sqrt(ndv * ndv * (1.0 - a2) + a2) + ndv * sqrt(ndl * ndl * (1.0 - a2) + a2), 1e-30
  );
  let xv = 1.0 - ndv;
  let xv2 = xv * xv;
  let fresnel_v = f0 + (1.0 - f0) * (xv2 * xv2 * xv);
  var out = light.radiance * ((1.0 - fresnel_v) * c_diff * g);
  if (ndl > 0.0) {
    out += light.radiance * (PI * fresnel * distribution * vis * ndl * ah.weight);
  }
  return out;
}

struct FragmentOut {
  @location(0) color: vec4<f32>,
  @location(1) object: f32,
};

@fragment
fn fs_main(in: VertexOut, @builtin(front_facing) front: bool) -> FragmentOut {
  // Derivatives first, in uniform control flow, at pixel centers: every texture lookup
  // derives from them.
  let dbdx = dpdxFine(in.center);
  let dbdy = dpdyFine(in.center);
  let inst = instances[in.instance];
  // The outward side faces the camera when the screen winding is counter-clockwise,
  // or clockwise for mirrored instances. Two-sided lighting turns normals toward it.
  let toward = select(-1.0, 1.0, front != ((inst.flags & INSTANCE_MIRRORED) != 0u));
  let m = in.material;
  let flags = material_flags(m);
  if ((flags & MATERIAL_DOUBLE_SIDED) == 0u && toward < 0.0) {
    discard;  // single-sided material seen from behind
  }
  let face = in.face * toward;
  var n = face;
  var shadow_origin = in.position;
  if ((inst.flags & INSTANCE_HAS_NORMALS) != 0u) {
    let len = length(in.normal);
    if (len > 1e-6) {
      n = in.normal * (toward / len);
    }
    shadow_origin = smooth_surface_point(in, toward);
  }
  var vnormal = in.face;  // outward: the interpolated normal, or the face normal
  if ((inst.flags & INSTANCE_HAS_NORMALS) != 0u && length(in.normal) > 1e-6) {
    vnormal = in.normal;
  }
  let s = SurfaceInput(
    in.instance, in.barycentric, dbdx, dbdy, in.p0, in.p1, in.p2,
    in.uv01.xy, in.uv01.zw, in.uv2, n * toward, vnormal, in.tangent, in.color
  );
  let has_uvs = (inst.flags & INSTANCE_HAS_UVS) != 0u;
  let mb = material_base(m);
  var sv = evaluate_surface(m, s, has_uvs, false);
  if ((flags & MATERIAL_MASK) != 0u && sv.alpha < wordf(mb + 5u)) {
    discard;  // alpha cut-out
  }
  if (slot_binding(m, SLOT_NORMAL) != NO_BINDING && binding_usable(slot_binding(m, SLOT_NORMAL), has_uvs)) {
    n = sv.normal * toward;
  }
  let links = vec2<u32>(bitcast<u32>(inst.normal[0].w), bitcast<u32>(inst.normal[1].w));
  var radiance: vec3<f32>;
  if ((flags & MATERIAL_STANDARD) != 0u) {
    radiance = shade_standard(sv, wordf(mb + 8u), in.position, n, shadow_origin, links);
  } else {
    radiance = shade_simple(sv.base, wordf(mb + 8u), wordf(mb + 9u), sv.occlusion, in.position, n, shadow_origin, links);
  }
  var color = radiance + sv.emissive;
  if (frame.atmo.fog.w > 0.5) {  // aerial perspective along the camera ray
    let dist = length(in.position);
    color = atmo_fog(frame.atmo, color, in.position / max(dist, 1e-30), dist);
  }
  var out: FragmentOut;
  out.color = vec4<f32>(color, 1.0);
  out.object = f32(inst.index + 1u);
  return out;
}
