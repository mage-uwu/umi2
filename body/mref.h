/* mref.h: reference motion clips for the body (written by tools/motions_to_ref.py). */
#ifndef MREF_H
#define MREF_H
#include "body.h"

typedef struct { char name[17]; int frames; float fps; Vec *root; Quat *q; /* frames x NB */ } Clip;
typedef struct { int n; Clip *c; } Motions;

int mref_load(const char *path, Motions *M);
const Clip *mref_find(const Motions *M, const char *name);
/* Pose at time t (s), looping; outputs root position, body world rotations, and joint rotations (child
 * relative to parent; rel[0] is the root's world rotation). */
void mref_sample(const Clip *c, double t, Vec *root, Quat *world, Quat *rel);
#endif
