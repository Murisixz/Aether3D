// Aether3D GPU renderer -- the shadow pass (appended to common.wgsl).
//
// Renders the depth of every triangle as seen from one directional light into
// one layer of the shadow-map array. Depth-only: there is no fragment stage.

struct ShadowPass {
  matrix: mat4x4<f32>,  // camera-relative world -> shadow clip (orthographic)
};

@group(0) @binding(0) var<uniform> shadow_pass: ShadowPass;

@vertex
fn vs_shadow(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> @builtin(position) vec4<f32> {
  let tri = triangle_vertices(vertex / 3u);
  let p = mesh_position(tri[vertex % 3u]);
  return shadow_pass.matrix * vec4<f32>(relative_position(instances[instance], p), 1.0);
}
