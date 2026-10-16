/* Reference PNG unfilter (8-bit samples): rows of (1 + stride) bytes -> rows of stride bytes. */
#include <stdint.h>
#include <stdlib.h>
static inline uint8_t paeth(int a, int b, int c) {
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return (uint8_t)((pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c));
}
int unfilter(int64_t height, int64_t stride, int64_t bpp, const uint8_t *in, uint8_t *out) {
    const uint8_t *prev = NULL;
    for (int64_t r = 0; r < height; r++) {
        const uint8_t *f = in + r * (stride + 1) + 1; uint8_t *o = out + r * stride; int t = in[r * (stride + 1)];
        for (int64_t x = 0; x < stride; x++) {
            int a = x >= bpp ? o[x - bpp] : 0, b = prev ? prev[x] : 0, c = (prev && x >= bpp) ? prev[x - bpp] : 0, p;
            switch (t) { case 0: p = 0; break; case 1: p = a; break; case 2: p = b; break; case 3: p = (a + b) >> 1; break;
                         case 4: p = paeth(a, b, c); break; default: return -1; }
            o[x] = (uint8_t)(f[x] + p);
        }
        prev = o;
    }
    return 0;
}
