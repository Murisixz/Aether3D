/* Edge topology of triangle loops, numbered like EditableMesh._add_edge_use (first use order). */
#include <stdint.h>
#include <stdlib.h>
static inline uint64_t mix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; return k; }
/* loops (F,3) welded vertex ids; outputs: edge_of_use (3F), edge_lo/hi (E), face CSR (start E+1, faces 3F). Returns E. */
int64_t build_edges(int64_t F, const int64_t *loops, int64_t *edge_of_use, int64_t *edge_lo, int64_t *edge_hi,
                    int64_t *start, int64_t *faces) {
    int64_t cap = 1; while (cap < 6 * F) cap <<= 1;
    uint64_t *keys = malloc(cap * sizeof *keys); int64_t *vals = malloc(cap * sizeof *vals);
    for (int64_t i = 0; i < cap; i++) keys[i] = UINT64_MAX;
    int64_t E = 0;
    for (int64_t f = 0; f < F; f++) for (int k = 0; k < 3; k++) {
        int64_t a = loops[3 * f + k], b = loops[3 * f + (k + 1) % 3];
        int64_t lo = a < b ? a : b, hi = a < b ? b : a;
        uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
        uint64_t h = mix(key) & (cap - 1);
        while (keys[h] != UINT64_MAX && keys[h] != key) h = (h + 1) & (cap - 1);
        if (keys[h] == UINT64_MAX) { keys[h] = key; vals[h] = E; edge_lo[E] = lo; edge_hi[E] = hi; E++; }
        edge_of_use[3 * f + k] = vals[h];
    }
    free(keys); free(vals);
    for (int64_t e = 0; e <= E; e++) start[e] = 0;
    for (int64_t u = 0; u < 3 * F; u++) start[edge_of_use[u] + 1]++;
    for (int64_t e = 0; e < E; e++) start[e + 1] += start[e];
    int64_t *fill = malloc((E + 1) * sizeof *fill);
    for (int64_t e = 0; e < E; e++) fill[e] = start[e];
    for (int64_t u = 0; u < 3 * F; u++) faces[fill[edge_of_use[u]]++] = u / 3;
    free(fill);
    return E;
}
