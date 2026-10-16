//! E7: Aether3D's WGSL software traversal (`trace_ray`) vs hardware ray queries, same rays, same process.
use std::{fs, path::Path, time::Instant};
use wgpu::util::DeviceExt;

fn read_u32(p: &Path) -> Vec<u32> {
    fs::read(p).unwrap_or_else(|e| panic!("{p:?}: {e}")).chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_f32(p: &Path) -> Vec<f32> {
    read_u32(p).into_iter().map(f32::from_bits).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = Path::new(&args[1]);
    let scene = &args[2];
    let reps: usize = args.get(3).map_or(10, |s| s.parse().unwrap());
    let dir = data.join(scene);
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
    eprintln!("adapter: {} ({:?}), driver {} {}", info.name, info.backend, info.driver, info.driver_info);
    let feats = wgpu::Features::EXPERIMENTAL_RAY_QUERY | wgpu::Features::TIMESTAMP_QUERY;
    assert!(adapter.features().contains(feats), "missing features: {:?}", feats - adapter.features());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: feats,
        required_limits: adapter.limits(),
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .expect("device");
    let wait = || device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let u = wgpu::BufferUsages::empty();
    let pt: Vec<u32> = read_u32(&dir.join("pt.bin"));
    let mut params: Vec<u32> = read_u32(&dir.join("params.bin"));
    let pt_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("pt"), contents: bytemuck::cast_slice(&pt), usage: u | wgpu::BufferUsages::STORAGE });
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::cast_slice(&params), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });

    // Bottom levels (one per mesh) and the top level (one instance per scene instance).
    let mut sizes = vec![];
    let mut vbufs = vec![];
    let mut ibufs = vec![];
    let mut blases = vec![];
    for j in 0.. {
        let pp = dir.join(format!("mesh{j}_pos.bin"));
        if !pp.exists() { break; }
        let pos = read_f32(&pp);
        let idx = read_u32(&dir.join(format!("mesh{j}_idx.bin")));
        vbufs.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&pos), usage: wgpu::BufferUsages::BLAS_INPUT }));
        ibufs.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&idx), usage: wgpu::BufferUsages::BLAS_INPUT }));
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: (pos.len() / 3) as u32,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(idx.len() as u32),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        blases.push(device.create_blas(
            &wgpu::CreateBlasDescriptor { label: None, flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE, update_mode: wgpu::AccelerationStructureUpdateMode::Build },
            wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
        ));
        sizes.push(size);
    }
    let mesh_of = read_u32(&dir.join("inst_mesh.bin"));
    let xfm = read_f32(&dir.join("inst_xfm.bin"));
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        max_instances: mesh_of.len() as u32,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    for k in 0..mesh_of.len() {
        let t: [f32; 12] = xfm[12 * k..12 * k + 12].try_into().unwrap();
        *tlas.get_mut_single(k).unwrap() = Some(wgpu::TlasInstance::new(&blases[mesh_of[k] as usize], t, k as u32, 0xff));
    }
    let entries: Vec<wgpu::BlasBuildEntry> = (0..blases.len())
        .map(|j| wgpu::BlasBuildEntry {
            blas: &blases[j],
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &sizes[j],
                vertex_buffer: &vbufs[j],
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(&ibufs[j]),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        })
        .collect();
    wait();
    let t0 = Instant::now();
    let mut enc = device.create_command_encoder(&Default::default());
    enc.build_acceleration_structures(entries.iter(), std::iter::once(&tlas));
    queue.submit([enc.finish()]);
    wait();
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;

    let module = |name: &str| {
        let src = fs::read_to_string(data.join(format!("{name}.wgsl"))).unwrap();
        let t0 = Instant::now();
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name), layout: None, module: &m, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        wait();
        (p, t0.elapsed().as_secs_f64() * 1e3)
    };
    let (sw_pipe, sw_compile_ms) = module("sw");
    let (hw_pipe, hw_compile_ms) = module("hw");
    let qs = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 });
    let qbuf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let qread = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let period = queue.get_timestamp_period() as f64;
    println!("{{\"scene\": \"{scene}\", \"adapter\": \"{}\", \"as_build_ms\": {build_ms:.3}, \"sw_pipeline_ms\": {sw_compile_ms:.1}, \"hw_pipeline_ms\": {hw_compile_ms:.1}}}", info.name);
    for rname in ["camera", "bounce"] {
        let rays = read_f32(&dir.join(format!("rays_{rname}.bin")));
        let n = rays.len() / 8;
        params[7] = n as u32;
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));
        let rays_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&rays), usage: wgpu::BufferUsages::STORAGE });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 16) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        for (kname, pipe) in [("sw", &sw_pipe), ("hw", &hw_pipe)] {
            let first = if kname == "sw" { pt_buf.as_entire_binding() } else { wgpu::BindingResource::AccelerationStructure(&tlas) };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: first },
                    wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: rays_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                ],
            });
            let mut times = vec![];
            for rep in 0..reps + 2 {
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites { query_set: &qs, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }),
                    });
                    cp.set_pipeline(pipe);
                    cp.set_bind_group(0, &bg, &[]);
                    cp.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
                }
                enc.resolve_query_set(&qs, 0..2, &qbuf, 0);
                enc.copy_buffer_to_buffer(&qbuf, 0, &qread, 0, 16);
                if rep == reps + 1 {
                    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, (n * 16) as u64);
                }
                queue.submit([enc.finish()]);
                qread.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
                wait();
                let ts: Vec<u64> = qread.get_mapped_range(..).unwrap().chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
                qread.unmap();
                if rep >= 2 {
                    times.push((ts[1] - ts[0]) as f64 * period / 1e6);
                }
            }
            read_buf.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
            wait();
            fs::write(dir.join(format!("out_{rname}_{kname}.bin")), &*read_buf.get_mapped_range(..).unwrap()).unwrap();
            read_buf.unmap();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = times[times.len() / 2];
            println!(
                "{{\"scene\": \"{scene}\", \"rays\": \"{rname}\", \"n\": {n}, \"kernel\": \"{kname}\", \"median_ms\": {med:.4}, \"min_ms\": {:.4}, \"max_ms\": {:.4}, \"mrays_per_s\": {:.1}}}",
                times[0], times[times.len() - 1], n as f64 / med / 1e3
            );
        }
    }
}
