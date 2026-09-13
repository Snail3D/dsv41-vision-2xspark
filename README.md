# DeepSeek-V4.1-Flash (EXL3) WITH VISION on 2× DGX Spark

**The upstream 2×Spark recipe ships vision disabled. This repo enables it —
tested and working — via a documented kernel-contract discovery and a
bind-mounted patch set (no image rebuild).**

Based on [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark)
(same pack, same image, same `run.sh` flow). No image rebuild needed — every
patch binds into the running container.

## Results (verified on 2× DGX Spark, GB10, TP=2)

| Test | Result |
|---|---|
| Image description ("red square and yellow circle on blue background") | ✅ correct |
| Object counting (5 circles) | ✅ correct |
| No-text detection | ✅ correct |
| Text smoke test (recipe's `323` probe) | ✅ intact |
| Kernel errors after vision enable | 0 |
| Prose decode (200 tok, c=1, DSpark spec) | ~21 tok/s (unchanged) |
| First image turn | runs non-speculative (DSpark draft skips mm turns) |

## Why upstream couldn't do this (the kernel contract)

DeepSeek-V4.1's attention is dual-cache: every prefill row attends to a
sliding-window (128) set of candidates plus an indexer-selected compressed set.
FlashInfer's SM120 sparse-MLA prefill kernel (`sparse_mla_sm120_prefill.cu`)
enforces, for this model:

- **dual-lane** (compressed cache present — always true for V4.1): `topk == 128`
  exactly, per-row lengths via `sparse_topk_lens`/`extra_sparse_topk_lens`
- **single-lane** (no compressed cache): `topk ∈ {512, 1024, 2048}` — unreachable
  for V4.1, which always carries the compressed lane

vLLM's V4.1 vision code instead **inflates `topk`** by
`vision_max_n_token` (window + image budget = 1152/1024/512) — the dual-lane
kernel rejects every such call (`topk != 128 → return false`), which is why the
recipe author zeroed vision rather than reconcile the two interfaces.

**The working approximation:** keep the attention index at `topk = 128` (kernel
contract) and let the model's own **Lightning Indexer** carry image visibility —
image tokens are regular tokens in the context, the indexer scores them like
everything else, and its top-k lane surfaces the relevant ones. This is the
DSA design principle applied to the image-span problem: approximate the widened
bidirectional visibility through the indexer's selection instead of a widened
index matrix. Image preprocessing keeps a REAL token budget (896) so the
vision tower, encoder profile, and processor all behave normally.

Trade-off: in-image bidirectional visibility is not guaranteed for image spans
wider than the indexer's selection — small/medium images are effectively fully
visible; very large images lean on the indexer's relevance ranking.

**Second bug found in production (2026-09-13):** with vision enabled, the
config also silently enables the `mm_prefix` attention-metadata mode
(`is_mm_prefix_lm = vision_n_layers > 0`). That mode was never exercised at
long context by anyone (upstream ships text-only, where it auto-disables), and
a ~110K-token real session degraded mid-generation into multilingual garbage
(~1700 chars clean, then soup) with mm_prefix on. The patched sitecustomize now
sets `is_mm_prefix_lm = False` (+ the two related flags), making the attention
stack byte-identical to the proven text-only configuration while the vision
tower, encoder and image pipeline stay on. Image tokens flow as plain tokens
through the Lightning Indexer lane.

## The upstream bugs this fixes

(plus a fourth fix: `run.sh` never forwarded vision env to the worker rank,
so multi-node ranks silently ran different configs — head vision, worker text.)

1. **`sitecustomize.py` hardcodes `language_model_only=True`** — it zeroes
   `vision_max_n_token` on every config build, so requesting vision always
   crashed with `ValueError: math domain error` in
   `mm_preprocess.get_image_size_with_most_features` (sqrt of a negative
   budget). It could never load vision as shipped.
2. **`run.sh` never propagated vision env to the worker rank** — multi-node
   ranks silently disagreed on the config (head vision, worker text).
3. **Profile-dummy sizing read the zeroed attribute** — image preprocessing
   crashed even after (1); it needs a separate, real budget.

## How to apply

```bash
# from the sfxnz kit checkout (~/dsv41-exl3), after its own setup is done:
cp patches/sitecustomize.py       docker/patch/sitecustomize.py
cp patches/mm_preprocess_vision.py docker/patch/mm_preprocess_vision.py
cp patches/attention_vision.py     docker/patch/attention_vision.py
cp patches/sparse_swa_vision.py    docker/patch/sparse_swa_vision.py
cp patches/run.sh.vision           run.sh        # see PATCHES.md for the diff
cp patches/dsv41-launch.sh         dsv41-launch.sh
chmod +x run.sh dsv41-launch.sh
```

`run.sh.vision` = upstream `run.sh` + four additive changes (all behind env
flags, all documented in `PATCHES.md`):
- worker rank resolves its own `HF_CACHE` (boxes with different usernames)
- `DSV41_VISION` env forwarded to worker + container
- conditional bind-mounts of the four patched files (no image rebuild)
- optional `DSV41_PROBE` instrumentation (probe_fi.py) that logs the exact
  kernel-call contract — this is how the dual-lane interface was mapped

Launch (the launcher already sets these; shown for clarity):
```bash
LANGUAGE_MODEL_ONLY=0 DSV41_VISION=1 bash dsv41-launch.sh
```
The launcher carries CX7 fabric overrides (HEAD_IP / WORKER_HOST / IFACE /
HCA) — edit them for your net.

**Prereqs (learned the hard way — do not skip):**
- 64 GB swapfile on BOTH nodes (`fallocate -l 64G /swapfile; mkswap; swapon`,
  fstab-persistent). Weight load peaks ~124 GB on a 121 GB box; without swap
  the OOM killer eats the loader at 100% shards.
- Stop any keepalive service that force-restarts other engines (ours
  restarted a parked Qwen container mid-load twice).
- `docker rm -f` the old container before relaunch — `docker run` name
  conflicts fail silently and leave the stale engine running.

## Patch internals (PATCHES.md has the annotated diff)

- `sitecustomize.py`: config init sets `vision_max_n_token = 896` (real
  budget for preprocessing/processor/encoder sizing) AND
  `dsv41_attention_vmnt = 0` (attention index widening — the SM120 contract).
- `mm_preprocess_vision.py`: dummy-image sizing falls back to a real budget
  when the attention-facing attribute is clamped.
- `attention_vision.py` / `sparse_swa_vision.py`: the two attention-path
  readers of the widening switch to `dsv41_attention_vmnt` (=0), so prefill
  index rows stay at the 128-wide window and the kernel contract holds.
- Image visibility rides the Lightning Indexer top-k lane (512 wide, per-row
  lens), which is the DSv4 design intent: the indexer selects what matters.

## Verification method (reproduce the contract yourself)

`probe_fi.py` (enable `DSV41_PROBE=1`) monkeypatches the FlashInfer DSV4
entrypoint and logs every call's tensor shapes. Run it on a healthy text serve
and you get the working contract: `sparse_indices [N,128]` +
`extra_sparse_indices [N,512]` + per-row lens, dual-lane on the SWA and
compressed caches. Then compare against a vision call to see the inflation
that breaks the kernel. Evidence in `EVIDENCE.md`.

## Files

- `patches/` — the four bind-mounted patched files + instrumented `run.sh` +
  launcher
- `PATCHES.md` — annotated diffs vs upstream recipe
- `RUNBOOK.md` — day-2 operations (restart, rollback, gotchas)
- `EVIDENCE.md` — probe output + test transcripts
- `TP4-SAUCE.md` — deployment notes for the 3/4-box native-MXFP4 SGLang
  path (vision there is native and uncompromised; this repo's checkpoint
  download doubles as its prerequisite)

## Upstream momentum (2026-09-13)

The sfxnz recipe moved since our base: CUDA graphs are now the default
(`DSV41_ALLOW_CUDA_GRAPHS=1`, gated warmup replaces `ENFORCE_EAGER=1` —
measured ~+8% over eager p2b: 14.18 → 15.29 tok/s on their prose harness),
plus native p2b fused-MoE kernels (`widen_p2b_shapes.py`) and an Engram
prestage loader. Our PR [#3](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark/pull/3)
carries the vision patch set rebased onto that head; our deployed stack now
runs the new image (graphs on, eager off) with vision verified on top.

## Related work

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — the
  larger community Spark-cluster project; its B12X attention backend achieves
  native vision for the sibling model DeepSeek-V4-Flash (0731) on the same
  topology. No V4.1 recipe exists there yet (as of 2026-09-13).
- [bidual/awesome-dgx-spark](https://github.com/bidual/awesome-dgx-spark) —
  curated list of Spark clustering projects, including independent 2-node
  DeepSeek-V4 DSpark ports.

## License & credit

- Base recipe: sfxnz (repo link above) — follow its license terms
- Patches + documentation: MIT
- Model: DeepSeek V4.1-Flash, MIT; EXL3 pack by sfxnz (MIT)

No affiliation with NVIDIA, vLLM, FlashInfer, or DeepSeek. Provided as-is —
this touches attention metadata plumbing, so benchmark and spot-check outputs
on your own hardware before trusting it with anything important.
