#!/bin/bash
# Batch job for a GPU pod (run from a clone of the repo, the file server already started on :8000 over /w/out): climbing
# (pod_climb.sh from climb1, into /w/out/climb), then PARC parkour (pod_parc.sh from jump_stick, into /w/out/parc), one after
# the other on the one GPU; /w/out/DONE when both are.
G=/w/repo/physics_gym/gpu
OUT=/w/out/climb FROM=climb1 SECONDS_T=2400 bash $G/pod_climb.sh
OUT=/w/out/parc bash $G/pod_parc.sh
echo done > /w/out/DONE
