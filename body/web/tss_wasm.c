/* tss_wasm.c: the nanotss body policy for the browser, built to WebAssembly from nanotss_blt.c itself
 * (its portable path, which matches the AVX-512 build's actions to float precision). The physics stays
 * in body.js; this module only maps an observation to the mean action, carrying the recurrent state.
 *
 *   clang --target=wasm32-wasi -O2 -mexec-model=reactor ... -o tss.wasm   (see build_wasm.sh)
 *
 * Exports: tss_buf(n) -> pointer to an n-byte buffer; tss_load(ptr, tss_len, head_len) loads the .tss and
 * .head files laid end to end there; tss_obs() -> pointer to OBS floats (raw observation, normalised here);
 * tss_step() -> pointer to ACT floats (mean action); tss_reset() clears the recurrent state.
 */
#define main nanotss_main
#include "../../nanotss_blt.c"
#undef main
#define EXPORT(n) __attribute__((export_name(n)))
static Model *M; static Gen *G; static real *E, *Win, *bin, *Wo, *bo; static float *mean, *sd, obs[1024], act[64]; static int OBS, ACT;
EXPORT("tss_buf") void *tss_buf(int n) { return malloc(n); }
EXPORT("tss_load") int tss_load(unsigned char *p, int tss_len, int head_len) {
    FILE *f = fmemopen(p, tss_len, "rb"); if (!f) return -1;
    /* load()'s steps, reading the .tss bytes from memory */
    char magic[4]; int rs; Cfg c; if (fread(magic, 1, 4, f) != 4 || fread(&rs, 4, 1, f) != 1 || rs != (int)sizeof(real) || !read_cfg(f, &c, 0)) return -2;
    int e[4] = {1, 0, 0, 0}; if (!memcmp(magic, "TFX1", 4) && fread(e, 4, 4, f) != 4) return -3;
    ssm_groups = e[0]; ssm_ltc = e[1] & 15; n_loops = (e[1] >> 4) > 1 ? (e[1] >> 4) : 1; ssm_qknorm = e[2] & 1; ssm_rope = (e[2] >> 1) & 1; pos_feat = (real)(e[3] / 1e6);
    M = model_new(&c, 1, 1); size_t z[N_FIELDS]; sizes(&c, z); real *pf[N_FIELDS]; fields(&M->P, pf);
    for (int i = 0; i < N_FIELDS; i++) if (fread(pf[i], sizeof(real), z[i], f) != z[i]) return -4;
    fclose(f); model_requantize(M); G = gen_new(&M->c);
    const unsigned char *h = p + tss_len; int dims[3];
    if (head_len < 16 || memcmp(h, "TSPH", 4)) return -5;
    memcpy(dims, h + 4, 12); if (dims[0] != c.d || dims[1] > 1024 || dims[2] > 64) return -6;
    OBS = dims[1]; ACT = dims[2]; const int d = c.d;
    const size_t ES = (size_t)d * OBS + d + (size_t)ACT * d + 2 * ACT;
    if ((size_t)head_len < 16 + ES * sizeof(real) + 8 * (size_t)OBS) return -7;
    E = malloc(ES * sizeof(real)); memcpy(E, h + 16, ES * sizeof(real));
    Win = E; bin = Win + (size_t)d * OBS; Wo = bin + d; bo = Wo + (size_t)ACT * d;
    mean = malloc(4 * OBS); sd = malloc(4 * OBS); memcpy(mean, h + 16 + ES * sizeof(real), 4 * OBS); memcpy(sd, h + 16 + ES * sizeof(real) + 4 * OBS, 4 * OBS);
    return d;
}
EXPORT("tss_obs") float *tss_obs(void) { return obs; }
EXPORT("tss_reset") void tss_reset(void) { gen_reset(&M->c, G); }
EXPORT("tss_step") float *tss_step(void) {
    const int d = M->c.d; real x0[512], u[512], r, o[1024];
    for (int i = 0; i < OBS; i++) { float z = (obs[i] - mean[i]) / sd[i]; o[i] = z > 5 ? 5 : z < -5 ? -5 : z; }
    for (int i = 0; i < d; i++) { real s = bin[i]; for (int j = 0; j < OBS; j++) s += Win[(size_t)i * OBS + j] * o[j]; x0[i] = s; }
    gen_core(M, G, x0, NULL); rmsnorm_fwd(G->x, M->P.norm_f, u, d, &r);
    for (int k = 0; k < ACT; k++) act[k] = (float)tanh(bo[k] + dotr(Wo + (size_t)k * d, u, d));
    return act;
}
