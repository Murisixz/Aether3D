// Aether3D GPU renderer -- the finalize pass (compute).
//
// One invocation per output pixel. It averages the pixel's samples
// (supersample^2 texels x msaa samples each) in linear light -- this is the
// MSAA/supersampling resolve -- applies exposure, encodes sRGB to 8 bits, and
// writes the auxiliary outputs: view depth and object index at a
// representative sample, coverage, and per-object visible-sample counts.
// The last two counters count the pixels with a NaN or +infinity in their
// linear value and the covered pixels, so renders that read back only the
// image can still check both.
//
// {{...}} placeholders select the multisampled or single-sample texture types;
// they are filled in by aether3d/rendering/gpu/shaders.py, never from user input.

struct Params {
  size: vec4<u32>,   // width, height, supersample, msaa
  depth: vec4<f32>,  // near, far, 1 = perspective / 0 = orthographic, exposure scale
};

@group(0) @binding(0) var color_texture: {{COLOR_TEXTURE}};
@group(0) @binding(1) var object_texture: {{COLOR_TEXTURE}};
@group(0) @binding(2) var depth_texture: {{DEPTH_TEXTURE}};
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> srgb_thresholds: array<f32, 256>;
@group(0) @binding(5) var<storage, read_write> out_image: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_linear: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_aux: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_counts: array<atomic<u32>>;

// 8-bit sRGB code of a linear value: the number of code boundaries at or below it.
// srgb_thresholds[c] is the smallest float32 whose sRGB value rounds to code c or
// more (computed in float64 on the CPU), so the result equals the CPU renderer's
// round(255 * srgb(clamp(v, 0, 1))) exactly, without depending on the GPU's pow().
fn encode_srgb(v: f32) -> u32 {
  var lo = 0u;
  var hi = 255u;
  for (var step = 0u; step < 8u; step++) {
    let mid = (lo + hi + 1u) / 2u;
    if (srgb_thresholds[mid] <= v) {
      lo = mid;
    } else {
      hi = mid - 1u;
    }
  }
  return lo;
}

fn object_id(value: f32) -> u32 {
  return u32(value + 0.5);
}

// NaN or +infinity, decided on the bits: shader compilers may assume floats are
// finite and fold float comparisons that would detect them.
fn nan_or_positive_infinity(v: f32) -> bool {
  let bits = bitcast<u32>(v);
  return (bits & 0x7f800000u) == 0x7f800000u
    && ((bits & 0x007fffffu) != 0u || (bits >> 31u) == 0u);
}

@compute @workgroup_size(8, 8)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = params.size.x;
  let height = params.size.y;
  if (gid.x >= width || gid.y >= height) {
    return;
  }
  let k = params.size.z;
  let m = params.size.w;
  var sum = vec3<f32>(0.0);
  var covered = 0u;
  var run_id = 0u;
  var run = 0u;
  for (var sy = 0u; sy < k; sy++) {
    for (var sx = 0u; sx < k; sx++) {
      let texel = vec2<i32>(vec2<u32>(gid.x * k + sx, gid.y * k + sy));
      for (var s = 0u; s < m; s++) {
        sum += textureLoad(color_texture, texel, {{SAMPLE}}).rgb;
        let id = object_id(textureLoad(object_texture, texel, {{SAMPLE}}).r);
        if (id != 0u) {
          covered += 1u;
          if (id == run_id) {
            run += 1u;
          } else {
            if (run > 0u) {
              atomicAdd(&out_counts[run_id - 1u], run);
            }
            run_id = id;
            run = 1u;
          }
        }
      }
    }
  }
  if (run > 0u) {
    atomicAdd(&out_counts[run_id - 1u], run);
  }

  // Representative sample: the center texel of the supersampled block (the CPU
  // renderer's choice), and its first MSAA sample.
  let rep = vec2<i32>(vec2<u32>(gid.x * k + k / 2u, gid.y * k + k / 2u));
  let rep_id = object_id(textureLoad(object_texture, rep, 0).r);
  var view_depth = bitcast<f32>(0x7f800000u);  // +inf: background
  if (rep_id != 0u) {
    let z = textureLoad(depth_texture, rep, 0);  // reversed depth: near 1, far 0
    let near = params.depth.x;
    let far = params.depth.y;
    if (params.depth.z > 0.5) {
      view_depth = near * far / (near + z * (far - near));
    } else {
      view_depth = far - z * (far - near);
    }
  }

  let total = f32(k * k * m);
  let value = sum / total * params.depth.w;
  let p = gid.y * width + gid.x;
  out_image[p] = encode_srgb(value.r) | (encode_srgb(value.g) << 8u) | (encode_srgb(value.b) << 16u);
  out_linear[3u * p] = value.r;
  out_linear[3u * p + 1u] = value.g;
  out_linear[3u * p + 2u] = value.b;
  out_aux[3u * p] = view_depth;
  out_aux[3u * p + 1u] = bitcast<f32>(i32(rep_id) - 1);
  out_aux[3u * p + 2u] = f32(covered) / total;
  let counters = arrayLength(&out_counts);
  if (nan_or_positive_infinity(value.r) || nan_or_positive_infinity(value.g)
      || nan_or_positive_infinity(value.b)) {
    atomicAdd(&out_counts[counters - 2u], 1u);
  }
  if (covered > 0u) {
    atomicAdd(&out_counts[counters - 1u], 1u);
  }
}
