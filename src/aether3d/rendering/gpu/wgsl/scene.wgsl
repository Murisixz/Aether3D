// Aether3D GPU renderer -- the scene pass (appended to common.wgsl).
//
// Vertex stage: camera-relative positions, world normals and the exact face
// normal of every triangle. Fragment stage: the Stage 5 shading model
// (docs/RENDERING.md), evaluated in linear light:
//
//   radiance = base * (ambient + sum(light * max(n.l, 0) * attenuation * shadow))
//            + sum(light * specular * max(n.h, 0)^shininess * attenuation * shadow)
//
// with two-sided lighting (normals turned toward the camera). Outputs are the
// linear radiance and the object index + 1 of every sample (0 = background).
//
// `surface()` is the only place where an object's look enters shading; a
// material system (Stage 7) replaces it without touching lighting or passes.

const MIN_LIGHT_DISTANCE: f32 = 1e-4;

struct Frame {
  view_proj: mat4x4<f32>,  // camera-relative world -> clip; depth is reversed (near 1, far 0)
  forward: vec4<f32>,      // xyz: camera forward; w: 1 perspective, 0 orthographic
  ambient: vec4<f32>,      // rgb: linear ambient light (color x intensity)
  light_count: u32,
  pcf_radius: i32,         // shadow filter radius in texels
  shadow_texel: f32,       // one shadow-map texel in texture coordinates
  _pad: u32,
};

struct Light {
  radiance: vec3<f32>,     // linear color x intensity
  kind: u32,               // 0 directional, 1 point
  vector: vec3<f32>,       // directional: direction of travel; point: camera-relative position
  range: f32,              // point: distance where the light fades out, 0 = no range
  falloff: u32,            // point: 1 inverse square, 0 none
  shadow: i32,             // shadow-map layer, -1 = casts no shadow
  _pad0: u32,
  _pad1: u32,
};

struct ShadowMap {
  matrix: mat4x4<f32>,     // camera-relative world -> shadow clip (x, y in [-1, 1], depth in [0, 1])
  bias: f32,               // constant depth bias, shadow depth units
  normal_offset: f32,      // world units, scaled by the sine of the light angle
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(2) var<storage, read> lights: array<Light>;
@group(0) @binding(3) var<storage, read> shadow_maps: array<ShadowMap>;
@group(0) @binding(4) var shadow_texture: texture_depth_2d_array;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;

struct VertexOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) @interpolate(perspective, centroid) position: vec3<f32>,  // camera-relative world
  @location(1) @interpolate(perspective, centroid) normal: vec3<f32>,    // world, unit (or zero)
  @location(2) @interpolate(flat) face: vec3<f32>,                       // world face normal (outward)
  @location(3) @interpolate(flat) instance: u32,
  @location(4) @interpolate(perspective, centroid) barycentric: vec3<f32>,
  // The whole triangle (camera-relative corners and their world normals), for shadow lookups.
  @location(5) @interpolate(flat) p0: vec3<f32>,
  @location(6) @interpolate(flat) p1: vec3<f32>,
  @location(7) @interpolate(flat) p2: vec3<f32>,
  @location(8) @interpolate(flat) n0: vec3<f32>,
  @location(9) @interpolate(flat) n1: vec3<f32>,
  @location(10) @interpolate(flat) n2: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOut {
  let inst = instances[instance];
  let tri = triangle_vertices(vertex / 3u);
  let q0 = mesh_position(tri.x);
  let q1 = mesh_position(tri.y);
  let q2 = mesh_position(tri.z);
  let corner = vertex % 3u;
  var q = q0;
  if (corner == 1u) {
    q = q1;
  } else if (corner == 2u) {
    q = q2;
  }
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
  out.clip = frame.view_proj * vec4<f32>(position, 1.0);
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
  out.instance = instance;
  return out;
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

struct Surface {
  base_color: vec3<f32>,
  specular: f32,
  shininess: f32,
};

// The look of an object at one fragment (Stage 7: materials evaluate here).
fn surface(inst: Instance) -> Surface {
  return Surface(inst.base_color, inst.specular, inst.shininess);
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

// Linear radiance leaving `position` toward the camera; shadow lookups start at `shadow_origin`.
fn shade(surf: Surface, position: vec3<f32>, n: vec3<f32>, shadow_origin: vec3<f32>) -> vec3<f32> {
  var v = -frame.forward.xyz;
  if (frame.forward.w > 0.5) {
    v = safe_normalize(-position);
  }
  var diffuse = frame.ambient.rgb;
  var highlight = vec3<f32>(0.0);
  for (var i = 0u; i < frame.light_count; i++) {
    let light = lights[i];
    var l: vec3<f32>;
    var attenuation = 1.0;
    if (light.kind == 0u) {
      l = -light.vector;
    } else {
      let to_light = light.vector - position;
      let distance = length(to_light);
      l = to_light / max(distance, 1e-30);
      let d = max(distance, MIN_LIGHT_DISTANCE);
      if (light.falloff == 1u) {
        attenuation = 1.0 / (d * d);
      }
      if (light.range > 0.0) {
        let ratio = d / light.range;
        let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
        attenuation = attenuation * window * window;
      }
    }
    let ndl = dot(n, l);
    if (ndl <= 0.0) {
      continue;
    }
    var visibility = 1.0;
    if (light.shadow >= 0) {
      visibility = shadow_visibility(light.shadow, shadow_origin, n, l);
    }
    let weight = attenuation * visibility;
    diffuse += light.radiance * (ndl * weight);
    if (surf.specular > 0.0) {
      let h = l + v;
      let ndh = dot(n, h) / max(length(h), 1e-30);
      if (ndh > 0.0) {
        highlight += light.radiance * (surf.specular * pow(ndh, surf.shininess) * weight);
      }
    }
  }
  return surf.base_color * diffuse + highlight;
}

struct FragmentOut {
  @location(0) color: vec4<f32>,
  @location(1) object: f32,
};

@fragment
fn fs_main(in: VertexOut, @builtin(front_facing) front: bool) -> FragmentOut {
  let inst = instances[in.instance];
  // The outward side faces the camera when the screen winding is counter-clockwise,
  // or clockwise for mirrored instances. Two-sided lighting turns normals toward it.
  let toward = select(-1.0, 1.0, front != ((inst.flags & INSTANCE_MIRRORED) != 0u));
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
  var out: FragmentOut;
  out.color = vec4<f32>(shade(surface(inst), in.position, n, shadow_origin), 1.0);
  out.object = f32(inst.index + 1u);
  return out;
}
