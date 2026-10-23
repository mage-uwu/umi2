/* body_x8.c: world_step for 8 worlds at once, one per lane of a 512-bit vector (AVX-512 doubles).
 *
 * The same equations as body.c, operation for operation, on structure-of-arrays state. Where body.c branches
 * per world (a constraint already satisfied, a corner above the ground, a push on one body) every lane
 * computes and a mask keeps the lanes the scalar code would have skipped exactly as they were; the rare large
 * rotations and joint errors that body.c sends to sin/cos/atan2 are computed per lane with the same library
 * calls. Built with -ffp-contract=off on both sides the result is bit-identical to world_step (check with
 * tools/bench_physics.c --x8); with contraction on they differ only by rounding.
 *
 * Worlds stepped together must share substeps, iters and h; everything else may differ per lane.
 */
#include "body.h"

typedef double vd __attribute__((vector_size(64)));
typedef long long vl __attribute__((vector_size(64)));
typedef struct { vd x, y, z; } V8;
typedef struct { vd w, x, y, z; } Q8;
typedef struct { V8 x, v, w, xprev, inv_inertia; Q8 q, qprev; vd inv_mass; } B8;

#define G_ACC 9.81
#define ASSIST_K 2700.0
#define ASSIST_KR 2000.0
static inline vd S(double a) { return (vd){a, a, a, a, a, a, a, a}; }
static inline vd sel(vl m, vd a, vd b) { return (vd)(((vl)a & m) | ((vl)b & ~m)); }
static inline V8 selv(vl m, V8 a, V8 b) { return (V8){sel(m, a.x, b.x), sel(m, a.y, b.y), sel(m, a.z, b.z)}; }
static inline Q8 selq(vl m, Q8 a, Q8 b) { return (Q8){sel(m, a.w, b.w), sel(m, a.x, b.x), sel(m, a.y, b.y), sel(m, a.z, b.z)}; }
static inline int any(vl m) { long long r = 0; for (int i = 0; i < 8; i++) r |= m[i]; return r != 0; }
#ifdef __AVX512F__
#include <immintrin.h>
static inline vd vsqrt8(vd a) { return (vd)_mm512_sqrt_pd((__m512d)a); }
#else
static inline vd vsqrt8(vd a) { vd r; for (int i = 0; i < 8; i++) r[i] = __builtin_sqrt(a[i]); return r; }
#endif
static inline vd vabs8(vd a) { return (vd)((vl)a & ~(vl)S(-0.0)); }
static inline V8 V(vd x, vd y, vd z) { return (V8){x, y, z}; }
static inline V8 add(V8 a, V8 b) { return V(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline V8 sub(V8 a, V8 b) { return V(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline V8 scl(V8 a, vd s) { return V(a.x * s, a.y * s, a.z * s); }
static inline vd dot(V8 a, V8 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V8 crs(V8 a, V8 b) { return V(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline vd len(V8 a) { return vsqrt8(dot(a, a)); }
static inline Q8 qm(Q8 a, Q8 b) {
    return (Q8){a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
static inline Q8 qc(Q8 a) { return (Q8){a.w, -a.x, -a.y, -a.z}; }
static inline V8 qr(Q8 q, V8 v) { V8 u = V(q.x, q.y, q.z), t = scl(crs(u, v), S(2)); return add(add(v, scl(t, q.w)), crs(u, t)); }
static inline V8 cv(Vec a) { return V(S(a.x), S(a.y), S(a.z)); }

static inline V8 invI(const B8 *b, V8 v) {
    V8 l = qr(qc(b->q), v);
    l = V(l.x * b->inv_inertia.x, l.y * b->inv_inertia.y, l.z * b->inv_inertia.z);
    return qr(b->q, l);
}
static void rot(B8 *b, V8 dw, vl m) {   /* body.c rotate_by on the lanes in m */
    const vd a2 = dot(dw, dw);
    const vd s = S(0.5) + a2 * (S(-1.0 / 48) + a2 * (S(1.0 / 3840) + a2 * (S(-1.0 / 645120) + a2 * S(1.0 / 185794560))));
    Q8 e = {S(1) + a2 * (S(-1.0 / 8) + a2 * (S(1.0 / 384) + a2 * (S(-1.0 / 46080) + a2 * S(1.0 / 10321920)))), dw.x * s, dw.y * s, dw.z * s};
    const vl big = (a2 >= S(1e-2)) & m;
    if (any(big)) for (int i = 0; i < 8; i++) if (big[i]) { Quat q = qexp(v3(dw.x[i], dw.y[i], dw.z[i])); e.w[i] = q.w; e.x[i] = q.x; e.y[i] = q.y; e.z[i] = q.z; }
    const Q8 q = qm(e, b->q); const vd r = S(1) / vsqrt8(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    b->q = selq(m, (Q8){q.w * r, q.x * r, q.y * r, q.z * r}, b->q);
}
static V8 qlog8(Q8 q) {   /* body.c qlog_fast */
    const vl neg = q.w < S(0);
    q = selq(neg, (Q8){-q.w, -q.x, -q.y, -q.z}, q);
    const vd s2 = q.x * q.x + q.y * q.y + q.z * q.z, s = vsqrt8(s2), t2 = s2 / (q.w * q.w);
    vd f = S(2) / q.w * (S(1) + t2 * (S(-1.0 / 3) + t2 * (S(1.0 / 5) + t2 * (S(-1.0 / 7) + t2 * (S(1.0 / 9) + t2 * (S(-1.0 / 11) + t2 * (S(1.0 / 13) + t2 * (S(-1.0 / 15) + t2 * S(1.0 / 17)))))))));
    const vl tiny = s < S(1e-12), big = ~(s < S(0.1) * q.w) & ~tiny;
    if (any(big)) for (int i = 0; i < 8; i++) if (big[i]) f[i] = 2 * atan2(s[i], q.w[i]) / s[i];
    return selv(tiny, V(S(2) * q.x, S(2) * q.y, S(2) * q.z), scl(V(q.x, q.y, q.z), f));
}
static vd genw(const B8 *b, V8 r, V8 n, V8 *k) { V8 rn = crs(r, n); *k = invI(b, rn); return b->inv_mass + dot(rn, *k); }

/* body.c solve_pos (cap < 0) on lanes m; b2 NULL: a fixed point. Returns the multiplier (0 off the mask). */
static vd spos(B8 *b1, B8 *b2, V8 r1, V8 r2, V8 n, vd C, vd alpha, vd h, vl m) {
    V8 k1, k2 = V(S(0), S(0), S(0)); const vd w1 = genw(b1, r1, n, &k1), w2 = b2 ? genw(b2, r2, n, &k2) : S(0), at = alpha / (h * h);
    m &= ~(w1 + w2 + at <= S(0));
    const vd dl = sel(m, -C / (w1 + w2 + at), S(0));
    V8 p = scl(n, dl);
    b1->x = selv(m, add(b1->x, scl(p, b1->inv_mass)), b1->x); rot(b1, scl(k1, dl), m);
    if (b2) { b2->x = selv(m, sub(b2->x, scl(p, b2->inv_mass)), b2->x); rot(b2, scl(k2, -dl), m); }
    return dl;
}
static void sang(B8 *b1, B8 *b2, V8 n, vd theta, vd alpha, vd h, vl m) {
    const V8 k1 = invI(b1, n), k2 = invI(b2, n);
    const vd w = dot(n, k1) + dot(n, k2), at = alpha / (h * h);
    const vd dl = -theta / (w + at);
    rot(b1, scl(k1, dl), m); rot(b2, scl(k2, -dl), m);
}
static void vimp(B8 *b, V8 r, V8 n, vd dv, vl m) {
    V8 k; const vd w = genw(b, r, n, &k); m &= ~(w <= S(0));
    const vd j = dv / w;
    b->v = selv(m, add(b->v, scl(n, j * b->inv_mass)), b->v); b->w = selv(m, add(b->w, scl(k, j)), b->w);
}
static inline V8 safe_unit(V8 D, vd C, vl m) { return scl(D, S(1) / sel(m, C, S(1))); }
static const double CORNER[NCORNER][3] = {{-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {-1, 1, 1}, {1, 1, 1}};
static V8 corner8(int i, int k) { const Vec hf = BODY_DEFS[i].half; return cv(v3(CORNER[k][0] * hf.x, CORNER[k][1] * hf.y, CORNER[k][2] * hf.z)); }

typedef struct {
    B8 b[NB]; Q8 target[NB]; vd tone[NB], kp_scale, kd_scale, mu, push_left, assist; V8 push_force, assist_x, assist_v; Q8 assist_q;
    vl push_on[NB];
} W8;

static void substep8(W8 *W, double hs, int iters) {
    const vd h = S(hs); const vl ALL = {-1, -1, -1, -1, -1, -1, -1, -1};
    vd cl[NB][NCORNER]; vl cm[NB][NCORNER]; int cany[NB][NCORNER];
    for (int i = 0; i < NB; i++) {
        B8 *b = W->b + i;
        b->v.y = b->v.y - h * S(G_ACC);
        const vl pm = W->push_on[i] & (W->push_left > S(0));
        b->v = selv(pm, add(b->v, scl(W->push_force, h * b->inv_mass)), b->v);
        b->xprev = b->x; b->x = add(b->x, scl(b->v, h));
        V8 wl = qr(qc(b->q), b->w), Il = V(wl.x / b->inv_inertia.x, wl.y / b->inv_inertia.y, wl.z / b->inv_inertia.z);
        V8 tq = crs(wl, Il); tq = V(-tq.x * b->inv_inertia.x, -tq.y * b->inv_inertia.y, -tq.z * b->inv_inertia.z);
        b->w = add(b->w, scl(qr(b->q, tq), h));
        b->qprev = b->q; rot(b, scl(b->w, h), ALL);
    }
    W->push_left = sel(W->push_left > S(0), W->push_left - h, W->push_left);
    for (int it = 0; it < iters; it++)
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; B8 *c = W->b + i, *p = W->b + d->parent; const BodyDef *pd = BODY_DEFS + d->parent;
        V8 rc = qr(c->q, cv(vsub(d->anchor, d->center))), rp = qr(p->q, cv(vsub(d->anchor, pd->center)));
        V8 D = sub(add(c->x, rc), add(p->x, rp)); vd C = len(D); vl m = C > S(1e-12);
        spos(c, p, rc, rp, safe_unit(D, C, m), C, S(0), h, m);
        Q8 tw = qm(p->q, W->target[i]); V8 e = qlog8(qm(c->q, qc(tw))); vd th = len(e);
        const vd kp = S(d->kp) * W->kp_scale * W->tone[i], alpha = S(1) / kp;
        if (d->hinge) {
            const Vec ax = d->hinge == 1 ? v3(1, 0, 0) : d->hinge == 2 ? v3(0, 1, 0) : v3(0, 0, 1);
            V8 aw = qr(p->q, cv(ax)); V8 off = sub(e, scl(aw, dot(e, aw))); vd to = len(off); vl mo = to > S(1e-12);
            sang(c, p, safe_unit(off, to, mo), to, S(0), h, mo);
            tw = qm(p->q, W->target[i]); e = qlog8(qm(c->q, qc(tw))); vd ta = dot(e, aw); vl ma = vabs8(ta) > S(1e-12);
            sang(c, p, selv(ta > S(0), aw, scl(aw, S(-1))), vabs8(ta), alpha, h, ma);
        } else { vl mt = th > S(1e-12); sang(c, p, safe_unit(e, th, mt), th, alpha, h, mt); }
    }
    const vl am = W->assist > S(0);
    if (any(am)) {
        B8 *p = W->b + PELVIS; V8 D = sub(p->x, W->assist_x); vd C = len(D); vl m = am & (C > S(1e-12));
        spos(p, NULL, V(S(0), S(0), S(0)), V(S(0), S(0), S(0)), safe_unit(D, C, m), C, S(1) / (W->assist * S(ASSIST_K)), h, m);
        V8 e = qlog8(qm(p->q, qc(W->assist_q))); vd th = len(e); vl mt = am & (th > S(1e-12));
        V8 n = safe_unit(e, th, mt); vd w = dot(n, invI(p, n)), at = S(1) / (W->assist * S(ASSIST_KR) * h * h);
        rot(p, invI(p, scl(n, -th / (w + at))), mt);
    }
    for (int i = 0; i < NB; i++) {
        const Vec hf = BODY_DEFS[i].half; B8 *b = W->b + i;
        const vl far = (b->x.y * b->x.y > S(vdot(hf, hf))) & (b->x.y > S(0));
        for (int k = 0; k < NCORNER; k++) cany[i][k] = 0;
        if (!any(~far)) continue;
        for (int k = 0; k < NCORNER; k++) {
            V8 r = qr(b->q, corner8(i, k)), pw = add(b->x, r);
            const vl m = ~far & (pw.y < S(0));
            cm[i][k] = m; cany[i][k] = any(m);
            if (!cany[i][k]) continue;
            cl[i][k] = spos(b, NULL, r, V(S(0), S(0), S(0)), V(S(0), S(-1), S(0)), -pw.y, S(0), h, m);
        }
    }
    for (int i = 0; i < NB; i++) {
        B8 *b = W->b + i;
        b->v = scl(sub(b->x, b->xprev), S(1) / h);
        Q8 dq = qm(b->q, qc(b->qprev)); V8 w = scl(V(dq.x, dq.y, dq.z), S(2) / h);
        b->w = selv(dq.w >= S(0), w, scl(w, S(-1)));
    }
    for (int i = 0; i < NB; i++) for (int k = 0; k < NCORNER; k++) {
        if (!cany[i][k]) continue;
        B8 *b = W->b + i; const vl m = cm[i][k]; V8 r = qr(b->q, corner8(i, k));
        V8 vp = add(b->v, crs(b->w, r));
        vimp(b, r, V(S(0), S(1), S(0)), -vp.y, m & (vp.y < S(0)));
        vp = add(b->v, crs(b->w, r)); V8 vt = V(vp.x, S(0), vp.z); vd s = len(vt); const vl ms = m & (s > S(1e-9));
        vd lim = W->mu * vabs8(cl[i][k]) / h, dv = sel(s < lim, s, lim);
        vimp(b, r, safe_unit(vt, s, ms), -dv, ms);
    }
    if (any(am)) {
        B8 *p = W->b + PELVIS; vd f = h * W->assist * S(20); f = sel(f < S(1), f, S(1));
        p->v = selv(am, add(p->v, scl(sub(W->assist_v, p->v), f)), p->v); p->w = selv(am, scl(p->w, S(1) - f), p->w);
    }
    for (int i = 1; i < NB; i++) {
        const BodyDef *d = BODY_DEFS + i; B8 *c = W->b + i, *p = W->b + d->parent;
        V8 dw = sub(c->w, p->w); vd m = len(dw); const vl mm = ~(m < S(1e-12));
        V8 n = safe_unit(dw, m, mm), kc = invI(c, n), kp = invI(p, n); vd weff = dot(n, kc) + dot(n, kp);
        vd f = h * S(d->kd) * W->kd_scale * W->tone[i] * weff; f = sel(f > S(1), S(1), f);
        const vd j = -f * m / sel(mm, weff, S(1));
        c->w = selv(mm, add(c->w, scl(kc, j)), c->w); p->w = selv(mm, sub(p->w, scl(kp, j)), p->w);
    }
}

void world_step_x8(World **Ws, int n) {
    W8 W; World *w0 = Ws[0];
    for (int l = 0; l < 8; l++) {
        const World *w = Ws[l < n ? l : 0];
        for (int i = 0; i < NB; i++) {
            const Body *b = w->b + i; B8 *B = W.b + i;
            B->x.x[l] = b->x.x; B->x.y[l] = b->x.y; B->x.z[l] = b->x.z; B->v.x[l] = b->v.x; B->v.y[l] = b->v.y; B->v.z[l] = b->v.z;
            B->w.x[l] = b->w.x; B->w.y[l] = b->w.y; B->w.z[l] = b->w.z; B->xprev.x[l] = b->xprev.x; B->xprev.y[l] = b->xprev.y; B->xprev.z[l] = b->xprev.z;
            B->q.w[l] = b->q.w; B->q.x[l] = b->q.x; B->q.y[l] = b->q.y; B->q.z[l] = b->q.z; B->qprev.w[l] = b->qprev.w; B->qprev.x[l] = b->qprev.x; B->qprev.y[l] = b->qprev.y; B->qprev.z[l] = b->qprev.z;
            B->inv_mass[l] = b->inv_mass; B->inv_inertia.x[l] = b->inv_inertia.x; B->inv_inertia.y[l] = b->inv_inertia.y; B->inv_inertia.z[l] = b->inv_inertia.z;
            const Quat t = w->target[i]; W.target[i].w[l] = t.w; W.target[i].x[l] = t.x; W.target[i].y[l] = t.y; W.target[i].z[l] = t.z;
            W.tone[i][l] = w->tone[i]; W.push_on[i][l] = w->push_body == i ? -1 : 0;
        }
        W.kp_scale[l] = w->kp_scale; W.kd_scale[l] = w->kd_scale; W.mu[l] = w->mu; W.push_left[l] = w->push_left; W.assist[l] = w->assist;
        W.push_force.x[l] = w->push_force.x; W.push_force.y[l] = w->push_force.y; W.push_force.z[l] = w->push_force.z;
        W.assist_x.x[l] = w->assist_x.x; W.assist_x.y[l] = w->assist_x.y; W.assist_x.z[l] = w->assist_x.z;
        W.assist_v.x[l] = w->assist_v.x; W.assist_v.y[l] = w->assist_v.y; W.assist_v.z[l] = w->assist_v.z;
        W.assist_q.w[l] = w->assist_q.w; W.assist_q.x[l] = w->assist_q.x; W.assist_q.y[l] = w->assist_q.y; W.assist_q.z[l] = w->assist_q.z;
    }
    for (int s = 0; s < w0->substeps; s++) substep8(&W, w0->h, w0->iters);
    for (int l = 0; l < n; l++) {
        World *w = Ws[l];
        for (int i = 0; i < NB; i++) {
            Body *b = w->b + i; const B8 *B = W.b + i;
            b->x = v3(B->x.x[l], B->x.y[l], B->x.z[l]); b->v = v3(B->v.x[l], B->v.y[l], B->v.z[l]); b->w = v3(B->w.x[l], B->w.y[l], B->w.z[l]);
            b->xprev = v3(B->xprev.x[l], B->xprev.y[l], B->xprev.z[l]);
            b->q = (Quat){B->q.w[l], B->q.x[l], B->q.y[l], B->q.z[l]}; b->qprev = (Quat){B->qprev.w[l], B->qprev.x[l], B->qprev.y[l], B->qprev.z[l]};
        }
        w->push_left = W.push_left[l]; w->t += w->h * w->substeps;
    }
}
