// Aether3D GPU renderer -- declarations shared by the scene and shadow passes.
//
// Meshes are read by "vertex pulling": draws are non-indexed with three
// vertices per triangle, and the vertex shader fetches the triangle's indices,
// positions and normals from storage buffers. Every vertex therefore knows its
// whole triangle, which gives exact face normals without duplicating vertices.
//
// Positions are "camera-relative world" coordinates: world coordinates minus
// the camera's eye point, computed on the CPU in float64 and uploaded as
// float32 (one model matrix per instance). Large scene coordinates therefore
// keep their precision near the camera.

const INSTANCE_MIRRORED: u32 = 1u;     // world determinant < 0: screen winding is reversed
const INSTANCE_HAS_NORMALS: u32 = 2u;  // the mesh has vertex normals

struct Instance {
  model: array<vec4<f32>, 3>,   // rows of the camera-relative model matrix (3 x 4)
  normal: array<vec4<f32>, 3>,  // rows of the normal matrix (inverse transpose), xyz
  base_color: vec3<f32>,        // linear-light base color
  specular: f32,                // Blinn-Phong highlight strength
  shininess: f32,               // Blinn-Phong exponent
  flags: u32,                   // INSTANCE_* bits
  index: u32,                   // index in RenderScene.instances (object index)
  _pad: u32,
};

@group(0) @binding(1) var<storage, read> instances: array<Instance>;

@group(1) @binding(0) var<storage, read> mesh_positions: array<f32>;  // 3 floats per vertex
@group(1) @binding(1) var<storage, read> mesh_normals: array<f32>;    // 3 floats per vertex
@group(1) @binding(2) var<storage, read> mesh_indices: array<u32>;    // 3 per triangle

fn mesh_position(vertex: u32) -> vec3<f32> {
  let b = 3u * vertex;
  return vec3<f32>(mesh_positions[b], mesh_positions[b + 1u], mesh_positions[b + 2u]);
}

fn mesh_normal(vertex: u32) -> vec3<f32> {
  let b = 3u * vertex;
  return vec3<f32>(mesh_normals[b], mesh_normals[b + 1u], mesh_normals[b + 2u]);
}

fn triangle_vertices(triangle: u32) -> vec3<u32> {
  let b = 3u * triangle;
  return vec3<u32>(mesh_indices[b], mesh_indices[b + 1u], mesh_indices[b + 2u]);
}

// Object space -> camera-relative world space.
fn relative_position(inst: Instance, p: vec3<f32>) -> vec3<f32> {
  let h = vec4<f32>(p, 1.0);
  return vec3<f32>(dot(inst.model[0], h), dot(inst.model[1], h), dot(inst.model[2], h));
}

// Object-space normal -> world space (inverse transpose; keeps normals outward under mirroring).
fn world_normal(inst: Instance, n: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(inst.normal[0].xyz, n), dot(inst.normal[1].xyz, n), dot(inst.normal[2].xyz, n));
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  let len = length(v);
  if (len > 0.0) {
    return v / len;
  }
  return vec3<f32>(0.0);
}
