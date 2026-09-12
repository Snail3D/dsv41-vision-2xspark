# Annotated diffs vs the sfxnz recipe

All changes are additive and env-gated. Upstream recipe files were captured
before patching (`run.sh.orig`, unpatched `sitecustomize.py` survives in git).

## 1. `sitecustomize.py` — the vision-killing bug + the split attribute

Upstream (docker/patch/sitecustomize.py):
```python
def _v41_cfg_init_text_only(self, *args, **kwargs):
    _v41_cfg_init(self, *args, **kwargs)
    self.vision_max_n_token = text_only_max_image_tokens(
        getattr(self, "vision_max_n_token", 0), True   # ← hardcoded True
    )
```
`language_model_only=True` is passed unconditionally, so EVERY config build —
including a vision-requested one — zeroes `vision_max_n_token`. Downstream,
`mm_preprocess.get_image_size_with_most_features` computes
`budget = vision_max_n_token - 1 = -1` and calls `sqrt((budget-2)/r + 0.25)`
→ `ValueError: math domain error` at engine init. Vision can never load.

Patched:
```python
self.vision_max_n_token = 896    # real budget: preprocessing, processor,
                                 # encoder profile sizing (896 = 1024-128, so
                                 # window+896 lands on a supported topk)
self.dsv41_attention_vmnt = 0    # attention index widening: SM120 kernel
                                 # contract (dual-lane prefill needs topk==128)
```

## 2. `attention_vision.py` + `sparse_swa_vision.py` — read the new attribute

The two attention-path readers of the widening (vllm/models/deepseek_v4_1/
attention.py ~line 240 and vllm/v1/attention/backends/mla/sparse_swa.py
~line 444) read `vision_max_n_token` to compute
`prefill_index_width = window + max_image_tokens`. Both now read
`dsv41_attention_vmnt` (=0), so:
- prefill index rows stay 128 wide → kernel dual-lane contract holds
- `sparse_swa` takes its original text paths everywhere (no span buffers)

Everything else in these files is untouched (diff is 2 lines each).

## 3. `mm_preprocess_vision.py` — preprocessing keeps a real budget

`get_image_size_with_most_features` sized the profile dummy image from the
same zeroed attribute (crash #1). Now:
```python
budget = max(getattr(hf_config, "vision_max_n_token", 0) or 0, 1024) \
         - (COMPRESS_PAD_TO - 1)
```
The dummy image sizes as if the budget were 1024; real images are processor-
budgeted at ≤895 tokens.

## 4. `run.sh.vision` — four additive changes

1. **Worker HF cache** (multi-user fix): the head passed its own
   `/home/<headuser>/.cache/huggingface` to the worker, which mkdir-failed on
   a box with a different username. Worker branch now overrides
   `HF_CACHE=$HOME/.cache/huggingface`.
2. **Vision env propagation**: `DSV41_VISION` (and `DSV41_PROBE`) added to the
   container env args AND to the SSH worker env string — without this, rank 1
   silently ran text-only config while rank 0 ran vision (config skew = load
   hang or crash).
3. **Conditional bind-mounts** (no image rebuild): when `DSV41_VISION=1` or
   `DSV41_PROBE=1`, run.sh mounts the four patched files over the baked copies:
   - sitecustomize.py → /usr/lib/python3.12/sitecustomize.py
   - mm_preprocess_vision.py → .../vllm/models/deepseek_v4_1/common/mm_preprocess.py
   - attention_vision.py → .../vllm/models/deepseek_v4_1/attention.py
   - sparse_swa_vision.py → .../vllm/v1/attention/backends/mla/sparse_swa.py
4. **Probe hook**: with `DSV41_PROBE=1`, probe_fi.py monkeypatches
   `flashinfer.mla._core.trtllm_batch_decode_sparse_mla_dsv4` and logs call
   signatures to $DSV41_PROBE_LOG. Ground truth for the contract table in
   EVIDENCE.md.

## 5. `dsv41-launch.sh`

Fabric overrides (our CX7 net: HEAD_IP/WORKER_HOST/IFACE/HCA — the recipe
defaults are someone else's) + `LANGUAGE_MODEL_ONLY=0 DSV41_VISION=1`.

## Kernel contract reference (from sparse_mla_sm120_prefill.cu, flashinfer 0.6.18)

```
dispatch_dsv4_dual(num_heads, topk, topk_extra, extra_page_block_size, ...):
    accepted: topk == 128 (both fulltile and per-row variants)
              fulltile: topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr
                        && topk_extra % 64 == 0 && extra_page_block_size in {64, 2}
              per-row:  topk_length / extra_topk_length tensors allowed
dispatch_dsv4_single(num_heads, topk, ...):
    accepted: topk in {128, 192, 256, 512, 1024, 2048}   (BF16 for <=256, FP8 above)
              only when extra_kv_cache is null — unreachable for V4.1
```

Rejected shapes we measured (all with the compressed lane present):
topk=1152/ex=0, topk=1024/ex=0, topk=512/ex=512 — all `Unsupported
sparse-MLA prefill configuration`.

Working contract (probe output from the healthy serve):
```
sparse_indices=[N,1,128]:int32  swa_topk_lens=[N]:int32      # lane 1: SWA window
extra_sparse_indices=[N,1,512]:int32  extra_sparse_topk_lens=[N]:int32
compressed_kv_cache=[P,64,1,584]:uint8                       # lane 2: indexer top-k
```
