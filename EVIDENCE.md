# Evidence

Hardware: 2× DGX Spark (GB10, SM121, 121.7 GiB UMA each), ConnectX-7 RoCE
point-to-point, TP=2. Image: dsv41-flash-exl3-sm121 (pinned
vllm/vllm-openai:deepseekv41-flash-0909 digest + sfxnz overlay). Pack:
sfxnz/DeepSeek-V4.1-Flash-EXL3 @ 2.0bpw-mcg (334 GB). vLLM 0.1.dev20904,
FlashInfer 0.6.18.

## Rejected kernel shapes (vision inflation, all with compressed lane present)

| attempt | vision budget | kernel topk | topk_extra | verdict |
|---|---|---|---|---|
| upstream behavior | 1024 | 1152 | 0 | `Unsupported sparse-MLA prefill configuration` |
| clamp #1 | 896 | 1024 | 0 | rejected (dual needs topk==128) |
| clamp #2 | 384 | 512 | 512 | rejected |

Root cause (read from sparse_mla_sm120_prefill.cu): dual-lane requires
topk==128 in both fulltile and per-row variants; single-lane (topk up to 2048)
requires extra_kv_cache == null, unreachable for V4.1.

## Working contract (probe, healthy text serve — DSV41_PROBE=1)

```
call#3 sparse_indices=tensor(N,1,128):int32  swa_topk_lens=tensor(N):int32
       extra_sparse_indices=tensor(N,1,512):int32  extra_sparse_topk_lens=[N]
       compressed_kv_cache=tensor(P,64,1,584):uint8  swa_kv_cache=uint8[.,64,1,584]
       query=tensor(N,32,512):bfloat16  sinks=tensor(32):float32
```
Two segments: SWA window (128) on the swa cache; indexer top-k (512, per-row
lens) on the compressed cache. This is the interface vision must respect.

## Vision verification (final engine, patches active)

1. Test image: blue 320×160 canvas, red rectangle, yellow ellipse.
   Prompt: "Describe this image in one short sentence. What shapes and colors?"
   → "A red square and a yellow circle sit on a blue background." (200, 11.3 s)
2. Test image: five red circles on gray.
   Prompt: "How many circles are in this image? Answer with just the number."
   → "5" (200)
3. Shapes image again, prompt: "Read the text in this image exactly."
   → "The image contains no text. It only shows a red square on the left and a
   yellow oval on the right, both set against a bl[ue background]" (200) —
   correctly refused to hallucinate text.
4. Text smoke (recipe probe): "323" ✅
5. Kernel errors across all vision requests: 0
6. Prose decode after vision enable: unchanged (~21 tok/s c=1, DSpark spec)
