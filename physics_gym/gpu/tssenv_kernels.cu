/* tssenv_kernels.cu: the body environments on the GPU, one thread per environment.
 *
 * Not a port: body.c, env.c and mref.c are compiled here as device code (their BHD / BCONST markers), so every
 * world runs the same XPBD solver, task rewards, resets and style prior as the CPU trainer, operation for
 * operation. Built with --fmad=false (the CPU side's -ffp-contract=off); what can still differ is the last bit
 * of a libm call (sin, cos, exp, atan2, pow), which the physics only reaches for large rotations.
 *
 * No torch here: plain C launchers (tss_*), bound to PyTorch tensors by tssenv_torch.cpp.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include "../body.c"
#include "../env.c"
#include "../mref.c"

/* Each thread copies its Env to local memory (interleaved across threads, so accesses coalesce), works on it,
 * and writes it back. */
__global__ void k_reset(Env *envs, const TaskCfg *cfg, int n, const uint64_t *seeds, const uint8_t *mask) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= n || (mask && !mask[e])) return;
    Env E; env_reset(&E, cfg, seeds[e]); envs[e] = E;
}
__global__ void k_obs(const Env *envs, int n, float *out) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= n) return;
    Env E = envs[e]; env_obs(&E, out + (size_t)e * OBS_DIM);
}
/* env_step for every environment; on a time-limit end also the observation reached (for bootstrapping) */
__global__ void k_step(Env *envs, int n, const float *act, float *rew, int *done, int *fell, int *steps, float *term_obs) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= n) return;
    Env E = envs[e];
    rew[e] = (float)env_step(&E, act + (size_t)e * ACT_DIM);
    done[e] = E.done; fell[e] = E.done && E.step < E.cfg.steps; steps[e] = E.step;
    if (E.done && !fell[e]) env_obs(&E, term_obs + (size_t)e * OBS_DIM);
    envs[e] = E;
}
/* env_style_prepare's features, frame by frame */
__global__ void k_style_feat(const Clip *c, float *f) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= c->frames) return;
    pose_features(c->q + (size_t)t * NB, f + (size_t)t * STYLE_FEAT);
}
/* nanotss_blt.c's urand() after tss_work_seed(seed): n values per sequence (the routing-credit stream) */
__global__ void k_urand(const uint64_t *seeds, int B, int n, float *out) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= B) return;
    uint64_t s = seeds[b] * 0x9E3779B97F4A7C15ULL + 1;
    for (int i = 0; i < n; i++) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        out[(size_t)b * n + i] = (float)((s * 2685821657736338717ULL) >> 11) * (float)(1.0 / 9007199254740992.0);
    }
}

/* constants as the device sees them (debugging the GPU build): per body mass share, kp, kd, hinge, parent, anchor,
 * centre, half extents; then ACT_SCALES */
__global__ void k_consts(double *o) {
    if (threadIdx.x || blockIdx.x) return; int k = 0;
    for (int i = 0; i < NB; i++) { const BodyDef *d = BODY_DEFS + i; o[k++] = d->mass_frac; o[k++] = d->kp; o[k++] = d->kd; o[k++] = d->hinge; o[k++] = d->parent;
        o[k++] = d->anchor.x; o[k++] = d->anchor.y; o[k++] = d->anchor.z; o[k++] = d->center.x; o[k++] = d->center.y; o[k++] = d->center.z; o[k++] = d->half.x; o[k++] = d->half.y; o[k++] = d->half.z; }
    for (int i = 0; i < NB; i++) o[k++] = ACT_SCALES[i];
}
/* one world_step of env e's world only (no actions, rewards or resets), for comparing the physics alone */
__global__ void k_world_step(Env *envs, int e) { if (threadIdx.x || blockIdx.x) return; Env E = envs[e]; world_step(&E.W); envs[e] = E; }
#define GRID(n) (((n) + 63) / 64), 64
extern "C" {
int tss_env_size(void) { return (int)sizeof(Env); }
int tss_clip_size(void) { return (int)sizeof(Clip); }
int tss_cfg_size(void) { return (int)sizeof(TaskCfg); }
void tss_dims(int *obs, int *act, int *hz, int *feat) { *obs = OBS_DIM; *act = ACT_DIM; *hz = CTRL_HZ; *feat = STYLE_FEAT; }
const char *tss_init(void) {   /* the deep call chains (resets run 75 world steps) need more than the default stack */
    cudaError_t r = cudaDeviceSetLimit(cudaLimitStackSize, 32768); return r == cudaSuccess ? NULL : cudaGetErrorString(r);
}
/* Clip structs on the device: name, frames, fps, and device arrays root (frames x 3) and q (frames x NB x 4), doubles */
void tss_make_clip(void *dst, const char *name, int frames, float fps, void *root, void *q) {
    Clip c; memset(&c, 0, sizeof c); strncpy(c.name, name, 16); c.frames = frames; c.fps = fps; c.root = (Vec *)root; c.q = (Quat *)q;
    cudaMemcpy(dst, &c, sizeof c, cudaMemcpyHostToDevice);
}
void tss_make_cfg(void *dst, int task, void *clip, double push_max, double sway, double arm_tone, double push_every, int steps,
                  double init_noise, int fixed_start, double assist, double act_mul, void *const *style, int nstyle, double style_start, double goal_cone, double goal_tops, int free_heading,
                  int chain, double chain_yaw) {
    TaskCfg t; memset(&t, 0, sizeof t);
    t.task = task; t.clip = (const Clip *)clip; t.push_max = push_max; t.sway = sway; t.arm_tone = arm_tone; t.push_every = push_every;
    t.steps = steps; t.init_noise = init_noise; t.fixed_start = fixed_start; t.assist = assist; t.act_mul = act_mul;
    for (int i = 0; i < nstyle && i < 8; i++) t.style[i] = (const Clip *)style[i];
    t.nstyle = nstyle; t.style_start = style_start; t.goal_cone = goal_cone; t.goal_tops = goal_tops; t.free_heading = free_heading; t.chain = chain; t.chain_yaw = chain_yaw;
    cudaMemcpy(dst, &t, sizeof t, cudaMemcpyHostToDevice);
}
/* the style table: clips (device Clip pointers, host-side frame counts) and feature buffers (frames x STYLE_FEAT) */
const char *tss_set_style(void *const *clips, const int *frames, void *const *feat, int n) {
    StyleClip t[16]; memset(t, 0, sizeof t);
    for (int i = 0; i < n && i < 16; i++) {
        k_style_feat<<<GRID(frames[i])>>>((const Clip *)clips[i], (float *)feat[i]);
        t[i].c = (const Clip *)clips[i]; t[i].f = (float *)feat[i]; t[i].n = frames[i];
    }
    cudaMemcpyToSymbol(style_clips, t, sizeof t); cudaMemcpyToSymbol(style_n, &n, sizeof n);
    cudaError_t r = cudaDeviceSynchronize(); return r == cudaSuccess ? NULL : cudaGetErrorString(r);
}
void tss_reset(void *envs, const void *cfg, int n, const void *seeds, const void *mask, cudaStream_t st) {
    k_reset<<<GRID(n), 0, st>>>((Env *)envs, (const TaskCfg *)cfg, n, (const uint64_t *)seeds, (const uint8_t *)mask);
}
void tss_obs(const void *envs, int n, float *out, cudaStream_t st) { k_obs<<<GRID(n), 0, st>>>((const Env *)envs, n, out); }
void tss_step(void *envs, int n, const float *act, float *rew, int *done, int *fell, int *steps, float *term_obs, cudaStream_t st) {
    k_step<<<GRID(n), 0, st>>>((Env *)envs, n, act, rew, done, fell, steps, term_obs);
}
void tss_urand(const void *seeds, int B, int n, float *out, cudaStream_t st) { k_urand<<<GRID(B), 0, st>>>((const uint64_t *)seeds, B, n, out); }
void tss_consts(double *o) { k_consts<<<1, 1>>>(o); }
void tss_world_step(void *envs, int e) { k_world_step<<<1, 1>>>((Env *)envs, e); }
__global__ void k_nav(int r, int pass) { const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c < NAV_N * NAV_N) nav_pass(r, pass, c); }
/* room r of the bank (room.json as ROOM_REC doubles per box: c, h, q, mu, soft, grip), filled as body.c room_set_bank
 * fills it; nb: the bank's size (rooms 0..nb-1 in use) */
const char *tss_set_room_bank(int r, const double *v, int n, int nb) {
    if (r < 0 || r >= ROOM_BANK || nb > ROOM_BANK || n < 0 || n > ROOM_MAX) return "room bank: too many rooms or boxes";
    static RoomBox t[ROOM_MAX];
    for (int j = 0; j < n; j++, v += ROOM_REC) {
        RoomBox *o = t + j; o->c.x = v[0]; o->c.y = v[1]; o->c.z = v[2]; o->h.x = v[3]; o->h.y = v[4]; o->h.z = v[5];
        o->q.w = v[6]; o->q.x = v[7]; o->q.y = v[8]; o->q.z = v[9]; o->mu = v[10]; o->soft = v[11]; o->grip = (int)v[12];
        o->rad = sqrt(o->h.x * o->h.x + o->h.y * o->h.y + o->h.z * o->h.z);
    }
    if (n) cudaMemcpyToSymbol(room_box, t, sizeof(RoomBox) * n, sizeof(RoomBox) * ROOM_MAX * r);
    cudaMemcpyToSymbol(room_n, &n, sizeof n, sizeof(int) * r);
    cudaMemcpyToSymbol(room_nb, &nb, sizeof nb);
    for (int pass = 0; pass < 3; pass++) k_nav<<<GRID(NAV_N * NAV_N), 0>>>(r, pass);   /* its nav grid, as room_set_bank builds it */
    cudaError_t e = cudaDeviceSynchronize(); return e == cudaSuccess ? NULL : cudaGetErrorString(e);
}
const char *tss_set_room(const double *v, int n) { return tss_set_room_bank(0, v, n, n ? 1 : 0); }
/* the hand-up assist, from the next step on: in the TaskCfg new episodes copy, and in every running environment */
__global__ void k_fall(int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) fall_make(i); }
const char *tss_fall_bank(void) {   /* the matched starts' falls, as env_fall_bank builds them (once) */
    int n; cudaMemcpyFromSymbol(&n, fall_n, sizeof n); if (n) return NULL;
    k_fall<<<GRID(FALL_N)>>>(FALL_N); n = FALL_N; cudaMemcpyToSymbol(fall_n, &n, sizeof n);
    cudaError_t r = cudaDeviceSynchronize(); return r == cudaSuccess ? NULL : cudaGetErrorString(r);
}
const char *tss_set_starts(const void *s, int n) {   /* the start bank (env.h START_N), FallState records from the host */
    if (n > START_N) n = START_N;
    cudaMemcpyToSymbol(start_bank, s, sizeof(FallState) * n); cudaMemcpyToSymbol(start_n, &n, sizeof n);
    cudaError_t r = cudaDeviceSynchronize(); return r == cudaSuccess ? NULL : cudaGetErrorString(r);
}
int tss_start_size(void) { return (int)sizeof(FallState); }
void tss_env_offsets(int *jumps, int *room) { *jumps = (int)offsetof(Env, jumps); *room = (int)(offsetof(Env, W) + offsetof(World, room)); }   /* fields read from a batch of Envs */
__global__ void k_set_assist(Env *envs, int n, double a) { const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e < n) envs[e].cfg.assist = a; }
void tss_cfg_set_assist(void *cfg, void *envs, int n, double a) {
    cudaMemcpy((char *)cfg + offsetof(TaskCfg, assist), &a, sizeof a, cudaMemcpyHostToDevice);
    k_set_assist<<<GRID(n), 0>>>((Env *)envs, n, a); cudaDeviceSynchronize();
}
const char *tss_error(void) { cudaError_t r = cudaGetLastError(); return r == cudaSuccess ? NULL : cudaGetErrorString(r); }
}
