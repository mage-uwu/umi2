/* tss_parity.c: the C side of the GPU trainer's parity check (parity.py).
 *
 * Runs nanotss_blt.c's own training path on a batch written by parity.py: sequences of input vectors x
 * (in place of ppo_tss's observation projection), each from a given recurrent state or from zero, through
 * layer_fwd / layer_bwd via the sequence hook, with the loss sum_t W_t . rmsnorm(x_out_t) * norm_f.
 * Writes the outputs, the input gradient, the summed parameter gradients and visited rows of one pass, then
 * the parameters after `steps` calls of tss_train_step (byte-mode Muon + Adam, as ppo_tss.c trains).
 *
 *   tss_parity <model.tss | init:SEED> in.bin out.bin lr steps [init_out.tss]
 */
#define main nanotss_main
#include "../../nanotss_blt.c"
#undef main

typedef struct { int B, L, *lens, *has0; float *X, *W, *H0, *HH0, *XO, *GX; size_t S_all, HH_all; real *hs, *hh; double loss; } Batch;

static real hook(void *user, int b, const Cfg *c, const Params *P, const Params *Q, Params *G, unsigned char *hit, int *ns, Work *w, const Codes *k) {
    Batch *B = user; const int d = c->d, L = B->lens[b], nl = c->n_layer, N = NODES(c), T = c->T;
    w->k = k;
    if (B->has0[b]) {
        memcpy(B->hs, B->H0 + (size_t)b * B->S_all, B->S_all * sizeof(real)); memcpy(B->hh, B->HH0 + (size_t)b * B->HH_all, B->HH_all * sizeof(real));
        w->carry_h = B->hs; w->carry_hist = B->hh;
    } else { w->carry_h = NULL; w->carry_hist = NULL; }
    real *x = w->x, *h = w->h; Tape *tp = w->tp;
    memcpy(x, B->X + (size_t)b * B->L * d, (size_t)L * d * sizeof(real));
    for (int l = 0; l < nl; l++) {
        if (!(l & 1)) carry_pre(c, w, l, h);
        layer_fwd(c, P, Q, l, L, x, h, tp + l, w);
        if (!(l & 1)) carry_post(c, w, l, L, h);
    }
    memcpy(B->XO + (size_t)b * B->L * d, x, (size_t)L * d * sizeof(real));
    real *g = w->g; memset(g, 0, sizeof(real) * (size_t)L * d); double loss = 0; real u[512];
    for (int t = 0; t < L; t++) {
        real r; rmsnorm_fwd(x + (size_t)t * d, P->norm_f, u, d, &r);
        const float *wt = B->W + ((size_t)b * B->L + t) * d;
        for (int i = 0; i < d; i++) loss += wt[i] * u[i];
        rmsnorm_bwd(x + (size_t)t * d, P->norm_f, r, wt, g + (size_t)t * d, G->norm_f, d);
    }
    real *lam = w->lam;
    for (int l = nl - 1; l >= 0; l--) {
        if (!(l & 1)) memset(lam, 0, sizeof(real) * (size_t)T * N * c->n * c->p);
        layer_bwd(c, P, Q, G, l, L, g, lam, hit + node_off(c, l), tp + l, w);
    }
    memcpy(B->GX + (size_t)b * B->L * d, g, (size_t)L * d * sizeof(real));
    w->carry_h = w->carry_hist = NULL;
    *ns = L; B->loss += loss;
    return (real)loss;
}

static void *rd(FILE *f, size_t n) { void *p = malloc(n ? n : 1); if (n && fread(p, 1, n, f) != n) { fprintf(stderr, "short input\n"); exit(1); } return p; }

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: tss_parity <model.tss|init:SEED> in.bin out.bin lr steps [init_out.tss]\n"); return 2; }
    FILE *fi = fopen(argv[2], "rb"); if (!fi) return 1;
    int hdr[2]; if (fread(hdr, sizeof hdr, 1, fi) != 1) return 1;
    const int Bn = hdr[0], L = hdr[1];
    Model *m;
    if (!strncmp(argv[1], "init:", 5)) {   /* ppo_tss.c's model: its globals, Cfg and seeded init */
        const int seed = atoi(argv[1] + 5);
        ssm_groups = 4; ssm_qknorm = 1; ssm_rope = 0; n_loops = 1;
        Cfg gc = {.vocab = 256, .d = 64, .n_layer = 4, .T = 4, .D = 4, .n = 16, .p = 16, .K = 3, .group = 64, .walks = 2, .temp = 1, .fw_mask = 0, .fff_gate = 1};
        m = model_new(&gc, 1, L); g_rng = (uint64_t)seed * 7919 + 88172645463325252ULL; params_init(&gc, &m->P); model_requantize(m);
    } else if (!(m = load(argv[1], 1, L))) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    if (argc > 6) save(m, argv[6]);
    byte_muon = 1; byte_muon_model = m;
    const Cfg *c = &m->c; const int d = c->d;
    Batch B = {.B = Bn, .L = L};
    B.S_all = (size_t)(c->n_layer / 2) * c->T * NODES(c) * c->n * c->p; B.HH_all = (size_t)c->n_layer * (c->K - 1) * d;
    B.lens = rd(fi, sizeof(int) * Bn); B.has0 = rd(fi, sizeof(int) * Bn);
    B.X = rd(fi, sizeof(float) * (size_t)Bn * L * d); B.W = rd(fi, sizeof(float) * (size_t)Bn * L * d);
    B.H0 = rd(fi, sizeof(float) * (size_t)Bn * B.S_all); B.HH0 = rd(fi, sizeof(float) * (size_t)Bn * B.HH_all); fclose(fi);
    B.XO = calloc((size_t)Bn * L * d, sizeof(float)); B.GX = calloc((size_t)Bn * L * d, sizeof(float));
    B.hs = malloc(B.S_all * sizeof(real)); B.hh = malloc(B.HH_all * sizeof(real));
    seq_hook = hook; seq_hook_user = &B;
    /* one pass of gradients, seeded as the first train step seeds each sequence */
    const size_t rows = node_rows(c); int ns;
    for (int b = 0; b < Bn; b++) { tss_work_seed(m->ws[0], 1ULL * 1000003ULL + (uint64_t)b); hook(&B, b, c, &m->P, &m->Q, m->Gt, m->hits, &ns, m->ws[0], m->k); }
    FILE *fo = fopen(argv[3], "wb"); if (!fo) return 1;
    fwrite(B.XO, sizeof(float), (size_t)Bn * L * d, fo); fwrite(B.GX, sizeof(float), (size_t)Bn * L * d, fo); fwrite(&B.loss, sizeof(double), 1, fo);
    size_t z[N_FIELDS]; sizes(c, z); real *gf[N_FIELDS], *pf[N_FIELDS]; fields(m->Gt, gf); fields(&m->P, pf);
    for (int f = 0; f < N_FIELDS; f++) { fwrite(gf[f], sizeof(real), z[f], fo); memset(gf[f], 0, z[f] * sizeof(real)); }
    fwrite(m->hits, 1, rows, fo); memset(m->hits, 0, rows);
    /* training steps on the same batch */
    const int steps = atoi(argv[5]);
    Hyper hp = {.lr = (real)atof(argv[4]), .beta1 = .9, .beta2 = .999, .eps = 1e-8, .weight_decay = 0, .clip = 1};
    for (int s = 1; s <= steps; s++) {
        hp.step = s; B.loss = 0;
        tss_train_step(c, &m->P, &m->Q, &m->G, m->hit, &m->M, &m->V, m->Gt, m->hits, m->ws, 1, NULL, NULL, Bn, L, &hp, m->k);
        fwrite(&B.loss, sizeof(double), 1, fo);
    }
    for (int f = 0; f < N_FIELDS; f++) fwrite(pf[f], sizeof(real), z[f], fo);
    return fclose(fo);
}
