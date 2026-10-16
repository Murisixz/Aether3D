/* Embree 4 reference: same triangles (float32, exactly the mesh positions) and rays (rounded to float32). */
#include <embree4/rtcore.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { struct RTCRayQueryContext ctx; const int64_t *skip_inst, *skip_tri; } SkipCtx;

static void skip_filter(const struct RTCFilterFunctionNArguments *args) {
    SkipCtx *c = (SkipCtx *)args->context;
    if (!c->skip_inst) return;
    for (unsigned i = 0; i < args->N; i++) {
        if (args->valid[i] == 0) continue;
        unsigned ray_id = ((unsigned *)args->ray)[10 * args->N + i];  /* RTCRayN_id */
        unsigned prim = ((unsigned *)args->hit)[5 * args->N + i];  /* RTCHitN_primID */
        unsigned inst = ((unsigned *)args->hit)[7 * args->N + i];  /* RTCHitN_instID level 0 */
        int64_t mine = (inst == RTC_INVALID_GEOMETRY_ID) ? 0 : (int64_t)inst;
        if (mine == c->skip_inst[ray_id] && (int64_t)prim == c->skip_tri[ray_id]) args->valid[i] = 0;
    }
}
RTCDevice em_device(void) { return rtcNewDevice("verbose=0"); }
RTCScene em_mesh(RTCDevice dev, int64_t nv, const float *pos, int64_t nt, const uint32_t *idx, int robust) {
    RTCScene s = rtcNewScene(dev);
    rtcSetSceneFlags(s, RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS | (robust ? RTC_SCENE_FLAG_ROBUST : 0));
    RTCGeometry g = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
    float *vb = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), nv);
    memcpy(vb, pos, 3 * sizeof(float) * nv);
    uint32_t *ib = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(uint32_t), nt);
    memcpy(ib, idx, 3 * sizeof(uint32_t) * nt);
    rtcCommitGeometry(g); rtcAttachGeometry(s, g); rtcReleaseGeometry(g);
    rtcCommitScene(s);
    return s;
}
/* Top level: instance k uses meshes[mesh_of[k]] with a row-major 3x4 float transform. */
RTCScene em_instances(RTCDevice dev, int64_t n, const int64_t *mesh_of, RTCScene *meshes, const float *xfm, int robust) {
    RTCScene s = rtcNewScene(dev);
    rtcSetSceneFlags(s, RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS | (robust ? RTC_SCENE_FLAG_ROBUST : 0));
    for (int64_t k = 0; k < n; k++) {
        RTCGeometry g = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(g, meshes[mesh_of[k]]);
        rtcSetGeometryTransform(g, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, xfm + 12 * k);
        rtcCommitGeometry(g); rtcAttachGeometryByID(s, g, (unsigned)k); rtcReleaseGeometry(g);
    }
    rtcCommitScene(s);
    return s;
}
void em_trace(RTCScene s, int64_t n, const double *o, const double *d, const int64_t *skip_inst, const int64_t *skip_tri,
              int instanced, double *t, int64_t *inst, int64_t *prim, int nthreads) {
    #pragma omp parallel num_threads(nthreads)
    {
        SkipCtx c; rtcInitRayQueryContext(&c.ctx); c.skip_inst = skip_inst; c.skip_tri = skip_tri;
        struct RTCIntersectArguments args; rtcInitIntersectArguments(&args);
        args.context = &c.ctx;
        if (skip_inst) { args.filter = skip_filter; args.flags |= RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER; }
        #pragma omp for schedule(dynamic, 512)
        for (int64_t r = 0; r < n; r++) {
            struct RTCRayHit rh;
            rh.ray.org_x = (float)o[3 * r]; rh.ray.org_y = (float)o[3 * r + 1]; rh.ray.org_z = (float)o[3 * r + 2];
            rh.ray.dir_x = (float)d[3 * r]; rh.ray.dir_y = (float)d[3 * r + 1]; rh.ray.dir_z = (float)d[3 * r + 2];
            rh.ray.tnear = 0.0f; rh.ray.tfar = INFINITY; rh.ray.mask = 0xFFFFFFFF; rh.ray.flags = 0; rh.ray.time = 0.0f;
            rh.ray.id = (unsigned)r;
            rh.hit.geomID = RTC_INVALID_GEOMETRY_ID; rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            rtcIntersect1(s, &rh, &args);
            if (rh.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                t[r] = rh.ray.tfar; prim[r] = rh.hit.primID; inst[r] = instanced ? (int64_t)rh.hit.instID[0] : 0;
            } else { t[r] = INFINITY; prim[r] = -1; inst[r] = -1; }
        }
    }
}
double em_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
