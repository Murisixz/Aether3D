inline fn localMatrix(p: [*]const f64, q: [*]const f64, s: [*]const f64, m: *[16]f64) void {
    const x = q[0]; const y = q[1]; const z = q[2]; const w = q[3];
    const r00 = 1 - 2 * (y * y + z * z); const r01 = 2 * (x * y - z * w); const r02 = 2 * (x * z + y * w);
    const r10 = 2 * (x * y + z * w); const r11 = 1 - 2 * (x * x + z * z); const r12 = 2 * (y * z - x * w);
    const r20 = 2 * (x * z - y * w); const r21 = 2 * (y * z + x * w); const r22 = 1 - 2 * (x * x + y * y);
    m.* = .{ r00 * s[0], r01 * s[1], r02 * s[2], p[0], r10 * s[0], r11 * s[1], r12 * s[2], p[1],
             r20 * s[0], r21 * s[1], r22 * s[2], p[2], 0, 0, 0, 1 };
}
export fn world_matrices(n: i64, parent: [*]const i64, pos: [*]const f64, quat: [*]const f64, scale: [*]const f64, out: [*]f64) void {
    var local: [16]f64 = undefined;
    var k: usize = 0;
    const count: usize = @intCast(n);
    while (k < count) : (k += 1) {
        localMatrix(pos + 3 * k, quat + 4 * k, scale + 3 * k, &local);
        const o = out + 16 * k;
        if (parent[k] < 0) { for (0..16) |t| o[t] = local[t]; continue; }
        const a = out + 16 * @as(usize, @intCast(parent[k]));
        for (0..4) |i| { for (0..4) |j| {
            o[4 * i + j] = a[4 * i] * local[j] + a[4 * i + 1] * local[4 + j] + a[4 * i + 2] * local[8 + j] + a[4 * i + 3] * local[12 + j];
        } }
    }
}
export fn noop() void {}
