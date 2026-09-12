#!/usr/bin/env bash
# V4.1-Flash EXL3 trial launcher (bs1 = head, bs2 = worker) — our fabric overrides.
set -x
cd ~/dsv41-exl3
export DSV41_PROBE=0
export HEAD_IP=10.0.0.1
export WORKER_HOST=spark2@10.0.0.2  # user@ip, passwordless ssh required
export IFACE=enp1s0f0np0
export HCA=rocep1s0f0
export LANGUAGE_MODEL_ONLY=0 DSV41_VISION=1; time ./run.sh
