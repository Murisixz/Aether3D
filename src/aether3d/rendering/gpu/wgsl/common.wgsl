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
const INSTANCE_HAS_UVS: u32 = 4u;      // the mesh has texture coordinates
const INSTANCE_HAS_GROUPS: u32 = 8u;   // the mesh has per-triangle face groups
const INSTANCE_HAS_COLORS: u32 = 16u;  // the mesh has vertex colours (after its UVs, below)

struct Instance {
  model: array<vec4<f32>, 3>,   // rows of the camera-relative model matrix (3 x 4)
  normal: array<vec4<f32>, 3>,  // rows of the normal matrix (inverse transpose), xyz
  base_color: vec3<f32>,        // linear-light base color
  specular: f32,                // Blinn-Phong highlight strength
  shininess: f32,               // Blinn-Phong exponent
  flags: u32,                   // INSTANCE_* bits
  index: u32,                   // index in RenderScene.instances (object index)
  colors: u32,                  // first float of the vertex colours in mesh_uvs
};

@group(0) @binding(1) var<storage, read> instances: array<Instance>;

@group(1) @binding(0) var<storage, read> mesh_positions: array<f32>;  // 3 floats per vertex
@group(1) @binding(1) var<storage, read> mesh_normals: array<f32>;    // 3 floats per vertex
@group(1) @binding(2) var<storage, read> mesh_indices: array<u32>;    // 3 per triangle
// 2 floats per vertex (UVs), then 4 per vertex (linear RGBA vertex colours) when the mesh
// has colours, from float Instance.colors on.
@group(1) @binding(3) var<storage, read> mesh_uvs: array<f32>;
@group(1) @binding(4) var<storage, read> mesh_groups: array<u32>;     // 1 per triangle
// MikkTSpace tangent (xyz) and bitangent sign (w) of every triangle corner, uploaded for
// meshes drawn with a UV normal map (aether3d/rendering/tangents.py); a dummy otherwise.
@group(1) @binding(5) var<storage, read> mesh_tangents: array<f32>;   // 4 per corner

fn mesh_position(vertex: u32) -> vec3<f32> {
  let b = 3u * vertex;
  return vec3<f32>(mesh_positions[b], mesh_positions[b + 1u], mesh_positions[b + 2u]);
}

fn mesh_normal(vertex: u32) -> vec3<f32> {
  let b = 3u * vertex;
  return vec3<f32>(mesh_normals[b], mesh_normals[b + 1u], mesh_normals[b + 2u]);
}

fn mesh_uv(vertex: u32) -> vec2<f32> {
  let b = 2u * vertex;
  return vec2<f32>(mesh_uvs[b], mesh_uvs[b + 1u]);
}

fn mesh_color(inst: Instance, vertex: u32) -> vec4<f32> {
  let b = inst.colors + 4u * vertex;
  return vec4<f32>(mesh_uvs[b], mesh_uvs[b + 1u], mesh_uvs[b + 2u], mesh_uvs[b + 3u]);
}

fn mesh_tangent(corner: u32) -> vec4<f32> {
  let b = 4u * corner;
  return vec4<f32>(mesh_tangents[b], mesh_tangents[b + 1u], mesh_tangents[b + 2u], mesh_tangents[b + 3u]);
}

// A tangent (object space) -> world space, unit, with its sign (negated when mirrored).
fn world_tangent(inst: Instance, t: vec4<f32>) -> vec4<f32> {
  let v = vec3<f32>(dot(inst.model[0].xyz, t.xyz), dot(inst.model[1].xyz, t.xyz), dot(inst.model[2].xyz, t.xyz));
  var w = select(1.0, -1.0, t.w < 0.0);
  if ((inst.flags & INSTANCE_MIRRORED) != 0u) {
    w = -w;
  }
  return vec4<f32>(safe_normalize(v), w);
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
