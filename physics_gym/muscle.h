/* muscle.h: written by tools/muscle_envelope.py. Per body, its joint's muscles: strength (largest torque, N m) and
 * slew (fastest rise of torque, N m/s), from published maximum torques scaled to 45 kg and the capture's inverse dynamics. */
static BCONST double MUSCLE_MAX[NB] = {0.0, 112.5, 75.0, 15.0, 30.0, 26.2, 30.0, 26.2, 188.41, 127.71, 71.2, 188.41, 127.71, 71.2};
static BCONST double MUSCLE_SLEW[NB] = {0.0, 1125.0, 750.0, 150.0, 300.0, 262.0, 300.0, 262.0, 1884.1, 1277.1, 712.0, 1884.1, 1277.1, 712.0};
