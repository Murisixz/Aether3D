//! Phase 5 prototype: Aether3D's GPU path tracer (`rendering/gpu/pathtrace.py`) replayed in
//! a Rust wgpu host, with its own software traversal (`sw.wgsl`) or with `trace_ray` on
//! hardware ray queries (`hw.wgsl`). Reads what `export.py` captured and dispatches exactly
//! what `render_scene` dispatches: the same buffers and bind groups, the same tiles and
//! samples per dispatch, the counters cleared before and read back after every dispatch,
//! then the pixel-center pass and the accumulation planes read back.
//!
//!     hwpt DIR render ORDER [--sample-offset S]   ORDER: e.g. sw,hw,hw,sw
//!     hwpt DIR agree
//!
//! `render` prints one JSON line per render (stage times in ms, GPU time from timestamp
//! queries, ray counters) and writes the planes of each variant's last render to
//! `planes_<variant>[_offset].bin`. `agree` traces the agreement kernel's rays with both
//! traversals and writes `agree.bin` (see `agree.wgsl`).
use std::{collections::HashMap, fs, path::Path, time::Instant};
use wgpu::util::DeviceExt;

fn read(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| panic!("{p:?}: {e}"))
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

struct Plan {
    width: u32,
    height: u32,
    samples: u32,
    per: u32,
    acc_planes: u64,
    counter_slots: u64,
    fallback_slot: usize,
    instances: u32,
    meshes: u32,
    words: [usize; 4], // first sample, sample count, mode, tile (x, y, w, h)
    modes: [u32; 2],   // MODE_PATHS, MODE_CENTER
    tiles: Vec<[u32; 4]>,
}

fn plan(dir: &Path) -> Plan {
    let text = fs::read_to_string(dir.join("plan.txt")).expect("plan.txt");
    let mut m: HashMap<String, Vec<u64>> = HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if let Some(k) = it.next() {
            m.insert(k.to_string(), it.map(|v| v.parse().unwrap()).collect());
        }
    }
    let one = |k: &str| m[k][0];
    let t = &m["tiles"];
    let tiles: Vec<[u32; 4]> = t[1..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| c.map(|v| v as u32))
        .collect();
    assert_eq!(tiles.len() as u64, t[0]);
    let w = &m["words"];
    Plan {
        width: one("width") as u32,
        height: one("height") as u32,
        samples: one("samples") as u32,
        per: one("per") as u32,
        acc_planes: one("acc_planes"),
        counter_slots: one("counter_slots"),
        fallback_slot: one("fallback_slot") as usize,
        instances: one("instances") as u32,
        meshes: one("meshes") as u32,
        words: [w[0] as usize, w[1] as usize, w[2] as usize, w[3] as usize],
        modes: [m["modes"][0] as u32, m["modes"][1] as u32],
        tiles,
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: String,
    period_ns: f64,
}

impl Gpu {
    fn new() -> Gpu {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .expect("adapter");
        let info = adapter.get_info();
        let feats = wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY;
        assert!(
            adapter.features().contains(feats),
            "{}: missing {:?}",
            info.name,
            feats - adapter.features()
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: feats,
            required_limits: adapter.limits(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))
        .expect("device");
        let period_ns = queue.get_timestamp_period() as f64;
        Gpu {
            device,
            queue,
            adapter: format!("{} ({}, {})", info.name, info.driver, info.driver_info),
            period_ns,
        }
    }

    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }
}

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn buffer_type(ty: wgpu::BufferBindingType) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

/// The bind group layouts of `shaders.py` ("pathtrace", "pathtrace-mesh"), plus the
/// acceleration structure of the hardware variant.
struct Layouts {
    scene: wgpu::BindGroupLayout,
    mesh: wgpu::BindGroupLayout,
    accel: wgpu::BindGroupLayout,
}

impl Layouts {
    fn new(device: &wgpu::Device) -> Layouts {
        let ro = buffer_type(wgpu::BufferBindingType::Storage { read_only: true });
        let rw = buffer_type(wgpu::BufferBindingType::Storage { read_only: false });
        let uniform = buffer_type(wgpu::BufferBindingType::Uniform);
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        Layouts {
            scene: layout(
                "pathtrace",
                &[
                    entry(1, ro),
                    entry(6, ro),
                    entry(7, ro),
                    entry(8, ro),
                    entry(9, uniform),
                    entry(10, rw),
                    entry(11, rw),
                ],
            ),
            mesh: layout(
                "pathtrace-mesh",
                &(0..5).map(|b| entry(b, ro)).collect::<Vec<_>>(),
            ),
            accel: layout(
                "accel",
                &[entry(
                    0,
                    wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                )],
            ),
        }
    }
}

/// A compiled kernel: its pipeline, whether it binds the acceleration structure, and the
/// seconds compiling took.
struct Kernel {
    pipeline: wgpu::ComputePipeline,
    hw: bool,
    compile_ms: f64,
}

fn kernel(gpu: &Gpu, layouts: &Layouts, dir: &Path, name: &str, entry_point: &str) -> Kernel {
    let src = fs::read_to_string(dir.join(format!("{name}.wgsl"))).expect("wgsl");
    let hw = name != "sw";
    let t0 = Instant::now();
    let groups: Vec<Option<&wgpu::BindGroupLayout>> = if hw {
        vec![
            Some(&layouts.scene),
            Some(&layouts.mesh),
            Some(&layouts.accel),
        ]
    } else {
        vec![Some(&layouts.scene), Some(&layouts.mesh)]
    };
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &groups,
            immediate_size: 0,
        });
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
    gpu.wait();
    Kernel {
        pipeline,
        hw,
        compile_ms: t0.elapsed().as_secs_f64() * 1e3,
    }
}

/// The exported scene, in host memory.
struct Inputs {
    scene_group: Vec<(u32, Vec<u8>)>, // binding, contents (instances, materials, texels, pt)
    params: Vec<u32>,
    mesh: Vec<Vec<u8>>,
    blas: Vec<(Vec<u8>, Vec<u8>)>, // per mesh: float32 xyz vertices, u32 indices
    inst_mesh: Vec<u32>,
    inst_xfm: Vec<f32>,
}

fn inputs(dir: &Path, plan: &Plan) -> Inputs {
    let f = |name: &str| read(&dir.join(format!("{name}.bin")));
    Inputs {
        scene_group: vec![
            (1, f("instances")),
            (6, f("materials")),
            (7, f("texels")),
            (8, f("scene")),
        ],
        params: words(&f("params")),
        mesh: ["positions", "normals", "indices", "uvs", "groups"]
            .iter()
            .map(|n| f(&format!("mesh_{n}")))
            .collect(),
        blas: (0..plan.meshes)
            .map(|j| (f(&format!("blas{j}_pos")), f(&format!("blas{j}_idx"))))
            .collect(),
        inst_mesh: words(&f("inst_mesh")),
        inst_xfm: words(&f("inst_xfm"))
            .into_iter()
            .map(f32::from_bits)
            .collect(),
    }
}

fn ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

/// The bottom levels (one per mesh with triangles) and the top level (one instance per
/// scene instance, its camera-relative transform, its index as custom data).
fn build_accel(gpu: &Gpu, inp: &Inputs) -> (wgpu::Tlas, Vec<wgpu::Blas>, usize) {
    let device = &gpu.device;
    let mut sizes = vec![];
    let mut bufs = vec![];
    let mut blases: Vec<Option<wgpu::Blas>> = vec![];
    for (pos, idx) in &inp.blas {
        let (nv, ni) = (pos.len() / 12, idx.len() / 4);
        if ni == 0 || nv == 0 {
            blases.push(None);
            sizes.push(None);
            bufs.push(None);
            continue;
        }
        let input = wgpu::BufferUsages::BLAS_INPUT;
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: pos,
            usage: input,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: idx,
            usage: input,
        });
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: nv as u32,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(ni as u32),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        blases.push(Some(device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: None,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        )));
        sizes.push(Some(size));
        bufs.push(Some((vb, ib)));
    }
    let n = inp.inst_mesh.len();
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        max_instances: n.max(1) as u32,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    let mut placed = 0;
    for k in 0..n {
        if let Some(blas) = &blases[inp.inst_mesh[k] as usize] {
            let t: [f32; 12] = inp.inst_xfm[12 * k..12 * k + 12].try_into().unwrap();
            *tlas.get_mut_single(k).unwrap() =
                Some(wgpu::TlasInstance::new(blas, t, k as u32, 0xff));
            placed += 1;
        }
    }
    let entries: Vec<wgpu::BlasBuildEntry> = (0..blases.len())
        .filter_map(|j| {
            let blas = blases[j].as_ref()?;
            let (vb, ib) = bufs[j].as_ref().unwrap();
            Some(wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: sizes[j].as_ref().unwrap(),
                        vertex_buffer: vb,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: Some(ib),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            })
        })
        .collect();
    let mut enc = device.create_command_encoder(&Default::default());
    enc.build_acceleration_structures(entries.iter(), std::iter::once(&tlas));
    gpu.queue.submit([enc.finish()]);
    gpu.wait();
    drop(entries);
    (tlas, blases.into_iter().flatten().collect(), placed)
}

const COUNTERS: [&str; 7] = [
    "camera", "bounce", "shadow", "ao", "center", "through", "limited",
];

/// One render (or one agreement pass): render_scene's buffers, dispatches and readbacks.
fn render(
    gpu: &Gpu,
    layouts: &Layouts,
    plan: &Plan,
    inp: &Inputs,
    k: &Kernel,
    sample_offset: u32,
    agree: bool,
) -> (String, Vec<u8>) {
    let device = &gpu.device;
    let queue = &gpu.queue;
    let n_pix = (plan.width * plan.height) as u64;
    let storage = wgpu::BufferUsages::STORAGE;
    let copy = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    // ---- upload (render_scene: buffers and bind groups) ----
    let t0 = Instant::now();
    let init = |contents: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents,
            usage,
        })
    };
    let scene_bufs: Vec<(u32, wgpu::Buffer)> = inp
        .scene_group
        .iter()
        .map(|(b, data)| (*b, init(data, storage)))
        .collect();
    let params_buf = init(
        &vec![0u8; 4 * inp.params.len()],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let accum = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accumulation"),
        size: plan.acc_planes * n_pix * 4,
        usage: storage | copy,
        mapped_at_creation: false,
    });
    let counters_size = 4 * (plan.counter_slots + plan.instances.max(1) as u64);
    let counters = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("counters"),
        size: counters_size,
        usage: storage | copy,
        mapped_at_creation: false,
    });
    let mesh_bufs: Vec<wgpu::Buffer> = inp.mesh.iter().map(|d| init(d, storage)).collect();
    let mut entries: Vec<wgpu::BindGroupEntry> = scene_bufs
        .iter()
        .map(|(b, buf)| wgpu::BindGroupEntry {
            binding: *b,
            resource: buf.as_entire_binding(),
        })
        .collect();
    for (binding, buf) in [(9, &params_buf), (10, &accum), (11, &counters)] {
        entries.push(wgpu::BindGroupEntry {
            binding,
            resource: buf.as_entire_binding(),
        });
    }
    let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layouts.scene,
        entries: &entries,
    });
    let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layouts.mesh,
        entries: &mesh_bufs
            .iter()
            .enumerate()
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: b as u32,
                resource: buf.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    queue.submit([]);
    gpu.wait();
    let upload_ms = ms(t0);

    // ---- acceleration structures (the hardware variant only) ----
    let t0 = Instant::now();
    let accel = k.hw.then(|| build_accel(gpu, inp));
    let group2 = accel.as_ref().map(|(tlas, _, _)| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layouts.accel,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(tlas),
            }],
        })
    });
    let as_ms = if k.hw { ms(t0) } else { 0.0 };

    // ---- dispatches ----
    let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let qbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let ts_at = counters_size.div_ceil(8) * 8;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: ts_at + 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut params = inp.params.clone();
    let [w_first, w_count, w_mode, w_tile] = plan.words;
    // One dispatch: its ray counters and its GPU time (ns).
    let mut dispatch = |mode: u32, tile: &[u32; 4], first: u32, count: u32| -> (Vec<u32>, f64) {
        params[w_first] = first;
        params[w_count] = count;
        params[w_mode] = mode;
        params[w_tile..w_tile + 4].copy_from_slice(tile);
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));
        let mut enc = device.create_command_encoder(&Default::default());
        enc.clear_buffer(&counters, 0, None);
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &qs,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            cp.set_pipeline(&k.pipeline);
            cp.set_bind_group(0, &group0, &[]);
            cp.set_bind_group(1, &group1, &[]);
            if let Some(g) = &group2 {
                cp.set_bind_group(2, g, &[]);
            }
            cp.dispatch_workgroups(tile[2].div_ceil(8), tile[3].div_ceil(8), 1);
        }
        enc.resolve_query_set(&qs, 0..2, &qbuf, 0);
        enc.copy_buffer_to_buffer(&counters, 0, &staging, 0, counters_size);
        enc.copy_buffer_to_buffer(&qbuf, 0, &staging, ts_at, 16);
        queue.submit([enc.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        gpu.wait();
        let result = {
            let data = staging.get_mapped_range(..).unwrap();
            let ts = &data[ts_at as usize..ts_at as usize + 16];
            let a = u64::from_le_bytes(ts[0..8].try_into().unwrap());
            let b = u64::from_le_bytes(ts[8..16].try_into().unwrap());
            (words(&data[..32]), b.wrapping_sub(a) as f64 * gpu.period_ns)
        };
        staging.unmap();
        result
    };
    let mut totals = [0u64; 8];
    let mut dispatches = 0u32;
    let mut add = |(c, ns): (Vec<u32>, f64), gpu_ns: &mut f64| {
        for (t, v) in totals.iter_mut().zip(c) {
            *t += v as u64;
        }
        *gpu_ns += ns;
        dispatches += 1;
    };
    let [mode_paths, mode_center] = plan.modes;
    let mut trace_ns = 0f64;
    let t0 = Instant::now();
    if !agree {
        let mut done = 0;
        while done < plan.samples {
            let count = plan.per.min(plan.samples - done);
            for tile in &plan.tiles {
                add(
                    dispatch(mode_paths, tile, sample_offset + done, count),
                    &mut trace_ns,
                );
            }
            done += count;
        }
    }
    let trace_ms = ms(t0);
    let mut center_ns = 0f64;
    let t0 = Instant::now();
    for tile in &plan.tiles {
        add(dispatch(mode_center, tile, 0, 1), &mut center_ns);
    }
    let center_ms = ms(t0);
    let (trace_gpu_ms, center_gpu_ms) = (trace_ns / 1e6, center_ns / 1e6);

    // ---- readback ----
    let t0 = Instant::now();
    let size = plan.acc_planes * n_pix * 4;
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&accum, 0, &read_buf, 0, size);
    queue.submit([enc.finish()]);
    read_buf.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    gpu.wait();
    let planes = read_buf.get_mapped_range(..).unwrap().to_vec();
    read_buf.unmap();
    let readback_ms = ms(t0);

    let name = if agree {
        "agree"
    } else if k.hw {
        "hw"
    } else {
        "sw"
    };
    let rays: Vec<String> = COUNTERS
        .iter()
        .zip(totals)
        .map(|(n, v)| format!("\"{n}\": {v}"))
        .collect();
    let placed = accel.as_ref().map_or(0, |(_, _, p)| *p);
    let total_ms = upload_ms + as_ms + trace_ms + center_ms + readback_ms;
    let line = format!(
        "{{\"variant\": \"{name}\", \"sample_offset\": {sample_offset}, \"adapter\": \"{}\", \
         \"upload_ms\": {upload_ms:.2}, \"as_build_ms\": {as_ms:.2}, \"trace_ms\": {trace_ms:.2}, \
         \"trace_gpu_ms\": {trace_gpu_ms:.2}, \"center_ms\": {center_ms:.2}, \
         \"center_gpu_ms\": {center_gpu_ms:.3}, \"readback_ms\": {readback_ms:.2}, \
         \"total_ms\": {total_ms:.2}, \"compile_ms\": {:.1}, \"dispatches\": {dispatches}, \
         \"tlas_instances\": {placed}, \"fallbacks\": {}, \"rays\": {{{}}}}}",
        gpu.adapter,
        k.compile_ms,
        totals[plan.fallback_slot],
        rays.join(", ")
    );
    (line, planes)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let what = args[2].as_str();
    let plan = plan(dir);
    let inp = inputs(dir, &plan);
    let gpu = Gpu::new();
    let layouts = Layouts::new(&gpu.device);
    match what {
        "render" => {
            let order: Vec<&str> = args[3].split(',').collect();
            let mut offset = 0;
            if let Some(i) = args.iter().position(|a| a == "--sample-offset") {
                offset = args[i + 1].parse().unwrap();
            }
            let sw = kernel(&gpu, &layouts, dir, "sw", "path_main");
            let hw = kernel(&gpu, &layouts, dir, "hw", "path_main");
            let mut last: HashMap<&str, Vec<u8>> = HashMap::new();
            for v in &order {
                let k = if *v == "hw" { &hw } else { &sw };
                let (line, planes) = render(&gpu, &layouts, &plan, &inp, k, offset, false);
                println!("{line}");
                last.insert(v, planes);
            }
            let suffix = if offset > 0 { "_offset" } else { "" };
            for (v, planes) in last {
                fs::write(dir.join(format!("planes_{v}{suffix}.bin")), planes).unwrap();
            }
        }
        "agree" => {
            let k = kernel(&gpu, &layouts, dir, "agree", "agree_main");
            let (line, planes) = render(&gpu, &layouts, &plan, &inp, &k, 0, true);
            println!("{line}");
            fs::write(dir.join("agree.bin"), planes).unwrap();
        }
        _ => panic!("usage: hwpt DIR render ORDER [--sample-offset S] | hwpt DIR agree"),
    }
}
