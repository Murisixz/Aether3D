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
  return LightSample(l, attenuation);
}

// The simple model (Aether3D 0.5): Lambert plus an optional Blinn-Phong highlight.
fn shade_simple(
  base: vec3<f32>, specular: f32, shininess: f32, occlusion: f32,
  position: vec3<f32>, n: vec3<f32>, shadow_origin: vec3<f32>
) -> vec3<f32> {
  let v = view_direction(position);
  var diffuse = frame.ambient.rgb * occlusion;
  var highlight = vec3<f32>(0.0);
  for (var i = 0u; i < frame.light_count; i++) {
    let light = lights[i];
    let ls = light_at(light, position);
    let ndl = dot(n, ls.l);
    if (ndl <= 0.0) {
      continue;
    }
    var visibility = 1.0;
    if (light.shadow >= 0) {
      visibility = shadow_visibility(light.shadow, shadow_origin, n, ls.l);
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
  sv: SurfaceValues, specular: f32, position: vec3<f32>, n: vec3<f32>, shadow_origin: vec3<f32>
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
  var out = frame.ambient.rgb * sv.occlusion * (c_diff + f0 * ab.x + ab.y);
  for (var i = 0u; i < frame.light_count; i++) {
    let light = lights[i];
    let ls = light_at(light, position);
    let ndl = dot(n, ls.l);
    if (ndl <= 0.0) {
      continue;
    }
    var visibility = 1.0;
    if (light.shadow >= 0) {
      visibility = shadow_visibility(light.shadow, shadow_origin, n, ls.l);
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

struct FragmentOut {
  @location(0) color: vec4<f32>,
  @location(1) object: f32,
};

@fragment
fn fs_main(in: VertexOut, @builtin(front_facing) front: bool) -> FragmentOut {
  // Derivatives first, in uniform control flow: every texture lookup derives from them.
  let dbdx = dpdxFine(in.barycentric);
  let dbdy = dpdyFine(in.barycentric);
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
  let s = SurfaceInput(
    in.instance, in.barycentric, dbdx, dbdy, in.p0, in.p1, in.p2,
    in.uv01.xy, in.uv01.zw, in.uv2, n * toward
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
  var radiance: vec3<f32>;
  if ((flags & MATERIAL_STANDARD) != 0u) {
    radiance = shade_standard(sv, wordf(mb + 8u), in.position, n, shadow_origin);
  } else {
    radiance = shade_simple(sv.base, wordf(mb + 8u), wordf(mb + 9u), sv.occlusion, in.position, n, shadow_origin);
  }
  var out: FragmentOut;
  out.color = vec4<f32>(radiance + sv.emissive, 1.0);
  out.object = f32(inst.index + 1u);
  return out;
}
