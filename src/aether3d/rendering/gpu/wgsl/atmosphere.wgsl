// Aether3D GPU renderer -- the outdoor atmosphere (scene.world.atmosphere): sky with a sun
// glow, exponential height fog with sun scattering, hemisphere ambient light. The same
// formulas as aether3d/rendering/atmosphere.py (the CPU renderers); see its docstring.

struct Atmo {
  flags: vec4<u32>,           // x: enabled, y: sky behind geometry, z: hemisphere ambient, w: environment (0 ambient, 1 uniform, 2 sky)
  sun: vec4<f32>,             // xyz: unit direction toward the sun; w: 1 with a sun
  zenith: vec4<f32>,          // rgb (linear, x intensity; uniform: the colour); w: glow size
  horizon: vec4<f32>,         // rgb; w: horizon glow
  ground: vec4<f32>,          // rgb; w: unused
  glow: vec4<f32>,            // rgb: glow colour x strength; w: fog sun power
  fog: vec4<f32>,             // x density, y falloff, z eye height above the fog height, w: 1 with fog
  fog_color: vec4<f32>,       // rgb
  fog_sun: vec4<f32>,         // rgb: sun colour x fog_sun
  ground_ambient: vec4<f32>,  // rgb: ground ambient x ambient intensity
  background: vec4<f32>,      // rgb: the world's background (environment "ambient")
  _pad: vec4<f32>,
};

fn atmo_sky(a: Atmo, d: vec3<f32>) -> vec3<f32> {
  let z = d.z;
  var base = a.zenith.rgb;
  if (a.flags.w == 2u) {
    if (z >= 0.0) {
      base = mix(a.horizon.rgb, a.zenith.rgb, sqrt(clamp(z, 0.0, 1.0)));
    } else {
      base = mix(a.horizon.rgb, a.ground.rgb, min(1.0, 4.0 * abs(z)));
    }
  } else if (a.flags.w == 0u) {
    base = a.background.rgb;
  }
  return base + atmo_glow(a, d);
}

// The sun glow and horizon glow added to the sky in direction `d`.
fn atmo_glow(a: Atmo, d: vec3<f32>) -> vec3<f32> {
  var base = vec3<f32>(0.0);
  let z = d.z;
  if (a.flags.x != 0u && a.sun.w > 0.5) {
    let g = max(dot(d, a.sun.xyz), 0.0);
    let size = a.zenith.w;
    let halo = pow(g, size) + 0.15 * pow(g, max(size / 8.0, 1.0));
    base += halo * a.glow.rgb;
    if (a.horizon.w > 0.0) {
      let sh = a.sun.xy;
      let sl = length(sh);
      let dl = length(d.xy);
      var c = 0.0;
      if (dl > 1e-9 && sl > 1e-9) {
        c = max(dot(d.xy, sh) / max(dl * sl, 1e-30), 0.0);
      }
      let t = 1.0 - min(2.5 * abs(z), 1.0);
      let band = t * t * t * t * (0.25 + 0.75 * c * c);
      let peak = max(max(a.glow.r, a.glow.g), max(a.glow.b, 1e-9));
      base += (a.horizon.w * band) * (a.glow.rgb / peak);
    }
  }
  return base;
}

// The fraction of light crossing the fog along the unit direction `d` over distance `t`
// (t < 0: toward the sky, an infinite distance).
fn atmo_transmittance(a: Atmo, d: vec3<f32>, t: f32) -> f32 {
  if (a.fog.w < 0.5) {
    return 1.0;
  }
  let b = a.fog.y;
  var rho = a.fog.x;
  if (b > 0.0) {
    rho = a.fog.x * exp(-b * a.fog.z);
  }
  var tau = 0.0;
  if (t < 0.0) {
    if (b > 0.0 && d.z > 1e-4) {
      tau = rho / (b * d.z);
    } else {
      tau = 1e30;
    }
  } else if (b > 0.0) {
    let x = b * d.z * t;
    if (abs(x) < 1e-4) {
      tau = rho * t * (1.0 - 0.5 * x);
    } else {
      tau = rho * (1.0 - exp(-x)) / (b * d.z);
    }
  } else {
    tau = rho * t;
  }
  return exp(-min(tau, 700.0));
}

// `color` seen through the fog along `d` over `t` (t < 0: the sky).
fn atmo_fog(a: Atmo, color: vec3<f32>, d: vec3<f32>, t: f32) -> vec3<f32> {
  if (a.fog.w < 0.5) {
    return color;
  }
  let trans = atmo_transmittance(a, d, t);
  var f = a.fog_color.rgb;
  if (a.sun.w > 0.5) {
    let g = max(dot(d, a.sun.xyz), 0.0);
    f += pow(g, a.glow.w) * a.fog_sun.rgb;
  }
  return color * trans + f * (1.0 - trans);
}

// The ambient light on a surface facing `n` (world, unit): the hemisphere blend of the
// world's ambient (up) and the ground ambient (down), or the uniform ambient.
fn atmo_ambient(a: Atmo, ambient: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  if (a.flags.x == 0u || a.flags.z == 0u) {
    return ambient;
  }
  let w = 0.5 + 0.5 * clamp(n.z, -1.0, 1.0);
  return mix(a.ground_ambient.rgb, ambient, w);
}
