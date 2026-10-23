/* policy.h: small recurrent policies trained by evolution strategies (no gradients needed).
 *
 *   mlp    obs -> tanh(64) -> tanh(64) -> tanh(act)                             (no memory; baseline)
 *   ssm    obs -> d, then SSM-tree blocks (as standalone/ssm_tree.c, streaming), -> tanh(act)
 *   ltc    liquid time-constant neurons (Hasani et al. 2020), fused semi-implicit Euler, unfolded
 *          UNFOLD times per control step:  x <- (x + dt f A) / (1 + dt (1/tau + f)),
 *          f = sigmoid(W x + U u + b) per neuron, then -> tanh(act)
 * Inputs are normalized with running mean/std held outside the parameter vector.
 */
#ifndef POLICY_H
#define POLICY_H
#include <stdint.h>

enum { POL_MLP, POL_SSM, POL_LTC };
typedef struct {
    int type, in, out;
    int hidden;                     /* mlp width, ltc neurons, ssm width d */
    int layers, trees, depth, n, p; /* ssm-tree shape */
} PolCfg;

typedef struct { float *h; float *x; } PolState;   /* recurrent state (ssm node states / ltc neurons) */

int pol_nparams(const PolCfg *c);
void pol_init(const PolCfg *c, float *theta, uint64_t seed);
PolState *pol_state_new(const PolCfg *c);
void pol_reset(const PolCfg *c, PolState *s);
void pol_act(const PolCfg *c, const float *theta, PolState *s, const float *obs_norm, float *act);
const char *pol_name(int type);
#endif
