# TP4 SAUCE — DeepSeek-V4.1-Flash (full precision) on 4× DGX Spark

*Emergency plan doc — written 2026-09-12 after the 2×Spark V4.1 swap-in.
Source repo (cloned alongside this file): MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks
(8 commits, 57 stars, actively iterated — re-check for updates before deploying).*

## What this is

The full-precision (native MXFP4 experts + FP8 dense) DeepSeek-V4.1-Flash served
from **4** DGX Spark (GB10) boxes over SGLang — no 2-bit quantization, ~2× the
decode speed of the 2×Spark EXL3 build, real concurrency headroom.

Measured by the repo authors (GB10 hardware, DSpark spec decode, prose):
- 1 stream: **37.9 tok/s**, TTFT 248 ms
- 2/3/4 streams: **58.9 / 71.2 / 78.6** agg, TTFT 424/311/383 ms
- Context: model max 1M; profile configures 1,048,576 with ~10 GB/rank slack

Compare (our fleet, 2026-09-12, 2×Spark EXL3 2.0bpw vLLM): 20.6 tok/s c=1,
48.5 agg at c=4 with 5.5 s TTFT.

## Why 4 boxes (not 2, not 3)

| Setup | resident weights/rank | native MXFP4 fits? | free mem/rank |
|---|---|---|---|
| 2× Spark (our current) | 145 GiB | NO → EXL3 2-bit pack | ~22 GB |
| 3× Spark (TP3) | ~101 GiB | yes (needs TP pad hacks: heads 64→96, draft experts 128→129) | ~6 GB |
| 4× Spark (TP4) | ~77 GiB | yes, all dims divide by 4 — **no padding** | ~40 GB |

TP4 is the clean config: no padded shards, 40 GB/rank free = real KV pool
(4M tokens), CUDA graphs for batch 1–8, 8 running requests.

## The model checkpoint

`deepseek-ai/DeepSeek-V4.1-Flash` (official, upstream) — 476 GiB on disk:
native MXFP4 experts (~305 GiB GPU-resident after Engram moves to NVMe) +
FP8 dense + two Engram n-gram tables (189 GiB → NVMe, ~63 GiB engram per node).

Backup copy downloaded to `~/Emergency-LLM/models/` (HF cache layout,
revision: main). Verify after download:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --cache-dir ~/Emergency-LLM/models   # resumable; re-run to verify
find ~/Emergency-LLM/models -name "*.safetensors" | wc -l   # expect ~48+
```

## Deploy steps (when the 4th Spark arrives)

1. **Network**: ConnectX-7 RoCE between all boxes (switch or pairwise CX7).
   On OUR fleet the fabric is `enp1s0f0np0` / HCA `rocep1s0f0`, IPs
   your-CX7-subnet — the repo's defaults are someone else's (`enp1s0f1np1`,
   10.0.0.x). Edit `.env.tp4` accordingly. Keep our Netplan `cx7-roce`
   profile discipline (see cx7-cluster skill; NetworkManager wipes RoCE IPs).
2. `cp -n .env.tp4.example .env.tp4` — set `HEAD_IP`, `WORKER_IPS`,
   `WORKER_HOSTS`, `WORKER_USER`, NFS server IPs, our fabric IFACE/HCA names.
3. `./start-tp4.sh doctor` — checks ssh, docker, IB, checkpoint shards, image
   arch, busy GPUs. Fix everything it flags.
4. `./start-tp4.sh build` — base image `lmsysorg/sglang:dev-dsv41` + overlay,
   on all four nodes.
5. `./start-tp4.sh share` — NFSv4 export of the checkpoint from the head
   (workers mount it; no rsync of 476 GiB).
6. `./start-tp4.sh pack` — engram shards `engram-l*-r<rank>of4.bin`,
   ~63 GiB per node, once.
7. `./start-tp4.sh serve` — API :8888 on the head. `status|logs|stop` subcommands
   exist; `logs worker3` targets a rank.

## Hard lessons we already paid for (apply them here)

- **Host RAM is GPU memory on GB10.** NCCL connection buffers are pinned RAM:
  the repo sets `NCCL_BUFFSIZE=1048576` etc. to cut 4.7→0.14 GiB pinned per
  node. Don't "optimize" these away.
- **Swapfile on every node** (64G, fstab) — our 2×Spark load OOMed at 124 GB
  peak on a 121 GB box; the same physics applies to 4-box loads. Check
  `swapon --show` on all boxes BEFORE first load.
- **Never OFFLOAD_MODE=ram** — Engram would evict the model.
- **Exclusive GPUs**: stop every other serving container first (our
  refuse-guards habit; this repo has `doctor` for it).
- **Keepalive services must be updated or stopped** — our loop-canary
  force-restarted the OLD engine mid-load and caused 2 OOM crashes. If any
  auto-restart service exists for the fleet, point it at the new container
  BEFORE the first load, not after.
- TP is synchronous: the slowest node paces all. Keep boxes identical,
  fabric clean, and monitor `free -g` (never nvidia-smi) on every node.

## Fallback chain (the "unfortunate scenario" ladder)

1. **Daily driver**: bs1+bs2, vLLM EXL3 2.0bpw TP2 — `~/dsv41-runbook.md` on bs1.
2. **4× Spark full precision**: this repo + `~/Emergency-LLM/models/` checkpoint.
3. **Gateway always**: snail-api on bs1:8900 / api.snail3d.com — re-point
   `SNAIL_FLASHNEXT_URL` at whichever engine is up (see env.bak files).
4. If all fleet engines are dead: MacBook 128 GB can't run V4.1, but MTPLX
   (`:8100`) + oMLX LilTop (`:8000`) lanes survive locally, and OpenRouter
   GLM-5.3-Flash (1M ctx) is the cloud fallback already configured in pi.

## Provenance / caveats

- Repo: 57★, 8 commits, 5 open issues, 7 open PRs at download time — young but
  active; benchmark numbers are the author's, not ours. Verify `doctor` output
  and re-bench on OUR fabric before trusting it.
- SGLang license mix: AGPL-3.0 + Apache-2.0 + MIT (see LICENSE* files).
- Our vLLM 2×Spark recipe remains the fallback and is the one already
  integrated with the gateway, pickers, and runbooks.