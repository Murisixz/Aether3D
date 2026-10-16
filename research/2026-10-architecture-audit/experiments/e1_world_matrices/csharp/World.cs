using System.Runtime.InteropServices;
public static unsafe class World {
    [UnmanagedCallersOnly(EntryPoint = "world_matrices")]
    public static void WorldMatrices(long n, long* parent, double* pos, double* quat, double* scale, double* outp) {
        double[] marr = new double[16]; fixed (double* m = marr) {
        for (long k = 0; k < n; k++) {
            double x = quat[4 * k], y = quat[4 * k + 1], z = quat[4 * k + 2], w = quat[4 * k + 3];
            double s0 = scale[3 * k], s1 = scale[3 * k + 1], s2 = scale[3 * k + 2];
            m[0] = (1 - 2 * (y * y + z * z)) * s0; m[1] = (2 * (x * y - z * w)) * s1; m[2] = (2 * (x * z + y * w)) * s2; m[3] = pos[3 * k];
            m[4] = (2 * (x * y + z * w)) * s0; m[5] = (1 - 2 * (x * x + z * z)) * s1; m[6] = (2 * (y * z - x * w)) * s2; m[7] = pos[3 * k + 1];
            m[8] = (2 * (x * z - y * w)) * s0; m[9] = (2 * (y * z + x * w)) * s1; m[10] = (1 - 2 * (x * x + y * y)) * s2; m[11] = pos[3 * k + 2];
            m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
            double* o = outp + 16 * k;
            long p = parent[k];
            if (p < 0) { for (int t = 0; t < 16; t++) o[t] = m[t]; continue; }
            double* a = outp + 16 * p;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
                o[4 * i + j] = a[4 * i] * m[j] + a[4 * i + 1] * m[4 + j] + a[4 * i + 2] * m[8 + j] + a[4 * i + 3] * m[12 + j];
        }
    } }
    [UnmanagedCallersOnly(EntryPoint = "noop")]
    public static void Noop() {}
}
