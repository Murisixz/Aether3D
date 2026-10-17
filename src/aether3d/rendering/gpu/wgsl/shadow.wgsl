// Aether3D GPU renderer -- the shadow pass (appended to common.wgsl and material.wgsl).
//
// Renders the depth of every triangle as seen from one directional light into
// one layer of the shadow-map array. Opaque instances are depth-only (no
// fragment stage). Instances with alpha-masked materials use vs_shadow_masked /
// fs_shadow_masked, which evaluate the material's opacity and discard cut-out
// fragments, so holes in leaves and fences let light through. Both sides of
// every triangle cast shadows, also for single-sided materials.

struct ShadowPass {
  matrix: mat4x4<f32>,  // camera-relative world -> shadow clip (orthographic)
  eye: vec4<f32>,       // xyz: the camera position in world space (for world coordinates)
};

@group(0) @binding(0) var<uniform> shadow_pass: ShadowPass;

fn camera_eye() -> vec3<f32> {
  return shadow_pass.eye.xyz;
}

@vertex
fn vs_shadow(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> @builtin(position) vec4<f32> {
  let tri = triangle_vertices(vertex / 3u);
  let p = mesh_position(tri[vertex % 3u]);
  return shadow_pass.matrix * vec4<f32>(relative_position(instances[instance], p), 1.0);
}

@vertex
fn vs_shadow_masked(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOut {
  return build_vertex(vertex, instance, shadow_pass.matrix);
}

@fragment
fn fs_shadow_masked(in: VertexOut) {
  let dbdx = dpdxFine(in.center);
  let dbdy = dpdyFine(in.center);
  let m = in.material;
  if ((material_flags(m) & MATERIAL_MASK) == 0u) {
    return;
  }
  let inst = instances[in.instance];
  var n = in.face;
  if ((inst.flags & INSTANCE_HAS_NORMALS) != 0u && length(in.normal) > 1e-6) {
    n = normalize(in.normal);
  }
  let s = SurfaceInput(
    in.instance, in.barycentric, dbdx, dbdy, in.p0, in.p1, in.p2, in.uv01.xy, in.uv01.zw, in.uv2, n,
    n, vec4<f32>(0.0)
  );
  let sv = evaluate_surface(m, s, (inst.flags & INSTANCE_HAS_UVS) != 0u, true);
  if (sv.alpha < wordf(material_base(m) + 5u)) {
    discard;
  }
}
