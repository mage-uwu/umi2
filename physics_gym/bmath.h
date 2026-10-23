/* bmath.h: sin, cos, exp, log, atan, atan2 for the body simulation, identical to the last bit on every build.
 *
 * The C libraries disagree in the last bit (glibc is close to correctly rounded, CUDA's device functions are
 * within 2 ulp), and in a contact-rich simulation one bit decides whether a corner touches the floor, so a GPU
 * build and a CPU build drift apart within an episode. These are fdlibm's algorithms (Sun Microsystems, 1993;
 * the same ones V8 uses for JavaScript's Math), written only with IEEE double operations, so with contraction
 * off (-ffp-contract=off, nvcc --fmad=false) every build returns the same bits. sqrt, floor and fmod need no
 * replacement: IEEE defines them exactly.
 *
 * Arguments beyond about 2^19 pi/2 for sin and cos (never reached here) fall back to the C library.
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
#ifndef BMATH_H
#define BMATH_H
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline BHD int32_t bm_hi(double x) { uint64_t u; memcpy(&u, &x, 8); return (int32_t)(u >> 32); }
static inline BHD uint32_t bm_lo(double x) { uint64_t u; memcpy(&u, &x, 8); return (uint32_t)u; }
static inline BHD double bm_words(int32_t hi, uint32_t lo) { uint64_t u = ((uint64_t)(uint32_t)hi << 32) | lo; double x; memcpy(&x, &u, 8); return x; }
static inline BHD double bm_set_hi(double x, int32_t hi) { return bm_words(hi, bm_lo(x)); }

static inline BHD double bm_kernel_sin(double x, double y, int iy) {
    const double S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03, S3 = -1.98412698298579493134e-04,
                 S4 = 2.75573137070700676789e-06, S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
    const int32_t ix = bm_hi(x) & 0x7fffffff;
    if (ix < 0x3e400000 && (int)x == 0) return x;
    const double z = x * x, v = z * x, r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if (iy == 0) return x + v * (S1 + z * r);
    return x - ((z * (0.5 * y - v * r) - y) - v * S1);
}
static inline BHD double bm_kernel_cos(double x, double y) {
    const double C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03, C3 = 2.48015872894767294178e-05,
                 C4 = -2.75573143513906633035e-07, C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
    const int32_t ix = bm_hi(x) & 0x7fffffff;
    if (ix < 0x3e400000 && (int)x == 0) return 1.0;
    const double z = x * x, r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if (ix < 0x3FD33333) return 1.0 - (0.5 * z - (z * r - x * y));
    const double qx = ix > 0x3fe90000 ? 0.28125 : bm_words(ix - 0x00200000, 0), hz = 0.5 * z - qx, a = 1.0 - qx;
    return a - (hz - (z * r - x * y));
}
/* x reduced by n pi/2 into y[0] + y[1]; returns n, or INT32_MIN for arguments this does not cover */
static inline BHD int bm_rem_pio2(double x, double *y) {
    const double invpio2 = 6.36619772367581382433e-01, pio2_1 = 1.57079632673412561417e+00, pio2_1t = 6.07710050650619224932e-11,
                 pio2_2 = 6.07710050630396597660e-11, pio2_2t = 2.02226624879595063154e-21, pio2_3 = 2.02226624871116645580e-21,
                 pio2_3t = 8.47842766036889956997e-32;
    const int32_t npio2_hw[32] = {0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB,
        0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A,
        0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB,
        0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB};
    const int32_t hx = bm_hi(x), ix = hx & 0x7fffffff;
    if (ix <= 0x3fe921fb) { y[0] = x; y[1] = 0; return 0; }
    if (ix < 0x4002d97c) {   /* |x| < 3pi/4: n = +-1 */
        double z;
        if (hx > 0) {
            z = x - pio2_1;
            if (ix != 0x3ff921fb) { y[0] = z - pio2_1t; y[1] = (z - y[0]) - pio2_1t; }
            else { z -= pio2_2; y[0] = z - pio2_2t; y[1] = (z - y[0]) - pio2_2t; }
            return 1;
        }
        z = x + pio2_1;
        if (ix != 0x3ff921fb) { y[0] = z + pio2_1t; y[1] = (z - y[0]) + pio2_1t; }
        else { z += pio2_2; y[0] = z + pio2_2t; y[1] = (z - y[0]) + pio2_2t; }
        return -1;
    }
    if (ix > 0x413921fb) return INT32_MIN;
    double t = fabs(x); const int n = (int)(t * invpio2 + 0.5); const double fn = (double)n;
    double r = t - fn * pio2_1, w = fn * pio2_1t;
    if (n < 32 && ix != npio2_hw[n - 1]) y[0] = r - w;
    else {
        const int32_t j = ix >> 20; y[0] = r - w;
        int32_t i = j - ((bm_hi(y[0]) >> 20) & 0x7ff);
        if (i > 16) {
            t = r; w = fn * pio2_2; r = t - w; w = fn * pio2_2t - ((t - r) - w); y[0] = r - w;
            i = j - ((bm_hi(y[0]) >> 20) & 0x7ff);
            if (i > 49) { t = r; w = fn * pio2_3; r = t - w; w = fn * pio2_3t - ((t - r) - w); y[0] = r - w; }
        }
    }
    y[1] = (r - y[0]) - w;
    if (hx < 0) { y[0] = -y[0]; y[1] = -y[1]; return -n; }
    return n;
}
static inline BHD double b_sin(double x) {
    const int32_t ix = bm_hi(x) & 0x7fffffff;
    if (ix <= 0x3fe921fb) return bm_kernel_sin(x, 0.0, 0);
    if (ix >= 0x7ff00000) return x - x;
    double y[2]; const int n = bm_rem_pio2(x, y);
    if (n == INT32_MIN) return sin(x);
    switch (n & 3) { case 0: return bm_kernel_sin(y[0], y[1], 1); case 1: return bm_kernel_cos(y[0], y[1]); case 2: return -bm_kernel_sin(y[0], y[1], 1); default: return -bm_kernel_cos(y[0], y[1]); }
}
static inline BHD double b_cos(double x) {
    const int32_t ix = bm_hi(x) & 0x7fffffff;
    if (ix <= 0x3fe921fb) return bm_kernel_cos(x, 0.0);
    if (ix >= 0x7ff00000) return x - x;
    double y[2]; const int n = bm_rem_pio2(x, y);
    if (n == INT32_MIN) return cos(x);
    switch (n & 3) { case 0: return bm_kernel_cos(y[0], y[1]); case 1: return -bm_kernel_sin(y[0], y[1], 1); case 2: return -bm_kernel_cos(y[0], y[1]); default: return bm_kernel_sin(y[0], y[1], 1); }
}
static inline BHD double b_exp(double x) {
    const double o_threshold = 7.09782712893383973096e+02, u_threshold = -7.45133219101941108420e+02, twom1000 = 9.33263618503218878990e-302,
                 ln2HI = 6.93147180369123816490e-01, ln2LO = 1.90821492927058770002e-10, invln2 = 1.44269504088896338700e+00,
                 P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03, P3 = 6.61375632143793436117e-05,
                 P4 = -1.65339022054652515390e-06, P5 = 4.13813679705723846039e-08;
    int32_t hx = bm_hi(x); const int xsb = (hx >> 31) & 1; hx &= 0x7fffffff;
    double hi = 0, lo = 0; int k = 0;
    if (hx >= 0x40862E42) {   /* |x| >= 709.78 */
        if (hx >= 0x7ff00000) return ((hx & 0xfffff) | bm_lo(x)) ? x + x : (xsb == 0 ? x : 0.0);
        if (x > o_threshold) return HUGE_VAL;
        if (x < u_threshold) return 0.0;
    }
    if (hx > 0x3fd62e42) {    /* |x| > ln2 / 2 */
        if (hx < 0x3FF0A2B2) { hi = xsb ? x + ln2HI : x - ln2HI; lo = xsb ? -ln2LO : ln2LO; k = 1 - xsb - xsb; }
        else { k = (int)(invln2 * x + (xsb ? -0.5 : 0.5)); const double t = k; hi = x - t * ln2HI; lo = t * ln2LO; }
        x = hi - lo;
    } else if (hx < 0x3e300000) return 1.0 + x;   /* |x| < 2^-28 */
    const double t = x * x, c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    if (k == 0) return 1.0 - ((x * c) / (c - 2.0) - x);
    const double y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if (k >= -1021) return bm_set_hi(y, bm_hi(y) + (k << 20));
    return bm_set_hi(y, bm_hi(y) + ((k + 1000) << 20)) * twom1000;
}
static inline BHD double b_log(double x) {
    const double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10, two54 = 1.80143985094819840000e+16,
                 Lg1 = 6.666666666666735130e-01, Lg2 = 3.999999999940941908e-01, Lg3 = 2.857142874366239149e-01, Lg4 = 2.222219843214978396e-01,
                 Lg5 = 1.818357216161805012e-01, Lg6 = 1.531383769920937332e-01, Lg7 = 1.479819860511658591e-01;
    int32_t hx = bm_hi(x); const uint32_t lx = bm_lo(x); int32_t k = 0;
    if (hx < 0x00100000) {    /* x < 2^-1022 */
        if (((hx & 0x7fffffff) | lx) == 0) return -HUGE_VAL;
        if (hx < 0) return NAN;
        k -= 54; x *= two54; hx = bm_hi(x);
    }
    if (hx >= 0x7ff00000) return x + x;
    k += (hx >> 20) - 1023; hx &= 0x000fffff;
    const int32_t i0 = (hx + 0x95f64) & 0x100000;
    x = bm_set_hi(x, hx | (i0 ^ 0x3ff00000)); k += i0 >> 20;
    const double f = x - 1.0;
    if ((0x000fffff & (2 + hx)) < 3) {   /* |f| < 2^-20 */
        if (f == 0.0) { if (k == 0) return 0.0; const double dk = (double)k; return dk * ln2_hi + dk * ln2_lo; }
        const double R = f * f * (0.5 - 0.33333333333333333 * f);
        if (k == 0) return f - R;
        const double dk = (double)k; return dk * ln2_hi - ((R - dk * ln2_lo) - f);
    }
    const double s = f / (2.0 + f), dk = (double)k, z = s * s, w = z * z;
    int32_t i = hx - 0x6147a; const int32_t j = 0x6b851 - hx;
    const double t1 = w * (Lg2 + w * (Lg4 + w * Lg6)), t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7))), R = t2 + t1;
    i |= j;
    if (i > 0) {
        const double hfsq = 0.5 * f * f;
        if (k == 0) return f - (hfsq - s * (hfsq + R));
        return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
    }
    if (k == 0) return f - s * (f - R);
    return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}
static inline BHD double b_atan(double x) {
    const double atanhi[4] = {4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01, 1.57079632679489655800e+00};
    const double atanlo[4] = {2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17};
    const double aT[11] = {3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01, -1.11111104054623557880e-01,
        9.09088713343650656196e-02, -7.69187620504482999495e-02, 6.66107313738753120669e-02, -5.83357013379057348645e-02,
        4.97687799461593236017e-02, -3.65315727442169155270e-02, 1.62858201153657823623e-02};
    const int32_t hx = bm_hi(x), ix = hx & 0x7fffffff; int id;
    if (ix >= 0x44100000) {   /* |x| >= 2^66 */
        if (ix > 0x7ff00000 || (ix == 0x7ff00000 && bm_lo(x) != 0)) return x + x;
        return hx > 0 ? atanhi[3] + atanlo[3] : -atanhi[3] - atanlo[3];
    }
    if (ix < 0x3fdc0000) {    /* |x| < 0.4375 */
        if (ix < 0x3e200000) return x;
        id = -1;
    } else {
        x = fabs(x);
        if (ix < 0x3ff30000) {
            if (ix < 0x3fe60000) { id = 0; x = (2.0 * x - 1.0) / (2.0 + x); } else { id = 1; x = (x - 1.0) / (x + 1.0); }
        } else {
            if (ix < 0x40038000) { id = 2; x = (x - 1.5) / (1.0 + 1.5 * x); } else { id = 3; x = -1.0 / x; }
        }
    }
    const double z = x * x, w = z * z;
    const double s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
    const double s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
    if (id < 0) return x - x * (s1 + s2);
    const double r = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);
    return hx < 0 ? -r : r;
}
static inline BHD double b_atan2(double y, double x) {
    const double tiny = 1.0e-300, pi_o_4 = 7.8539816339744827900E-01, pi_o_2 = 1.5707963267948965580E+00, pi = 3.1415926535897931160E+00,
                 pi_lo = 1.2246467991473531772E-16;
    const int32_t hx = bm_hi(x), ix = hx & 0x7fffffff, hy = bm_hi(y), iy = hy & 0x7fffffff;
    const uint32_t lx = bm_lo(x), ly = bm_lo(y);
    if ((uint32_t)(ix | ((lx | (0u - lx)) >> 31)) > 0x7ff00000u || (uint32_t)(iy | ((ly | (0u - ly)) >> 31)) > 0x7ff00000u) return x + y;
    if (((hx - 0x3ff00000) | (int32_t)lx) == 0) return b_atan(y);   /* x = 1 */
    const int m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
    if ((iy | (int32_t)ly) == 0) { switch (m) { case 0: case 1: return y; case 2: return pi + tiny; default: return -pi - tiny; } }
    if ((ix | (int32_t)lx) == 0) return hy < 0 ? -pi_o_2 - tiny : pi_o_2 + tiny;
    if (ix == 0x7ff00000) {
        if (iy == 0x7ff00000) { switch (m) { case 0: return pi_o_4 + tiny; case 1: return -pi_o_4 - tiny; case 2: return 3.0 * pi_o_4 + tiny; default: return -3.0 * pi_o_4 - tiny; } }
        switch (m) { case 0: return 0.0; case 1: return -0.0; case 2: return pi + tiny; default: return -pi - tiny; }
    }
    if (iy == 0x7ff00000) return hy < 0 ? -pi_o_2 - tiny : pi_o_2 + tiny;
    const int32_t k = (iy - ix) >> 20; double z;
    if (k > 60) z = pi_o_2 + 0.5 * pi_lo;
    else if (hx < 0 && k < -60) z = 0.0;
    else z = b_atan(fabs(y / x));
    switch (m) {
        case 0: return z;
        case 1: return -z;
        case 2: return pi - (z - pi_lo);
        default: return (z - pi_lo) - pi;
    }
}
static inline BHD double b_sq(double x) { return x * x; }   /* pow(x, 2) */
#endif
