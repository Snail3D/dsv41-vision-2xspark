# Day-2 operations

## Relaunch (normal)
Containers are `--restart=unless-stopped`; a node reboot brings the TP=2 pair
back automatically.

Full manual relaunch (after a wedge):
```bash
# worker first, then head — run.sh orchestrates this from the head node
docker rm -f dsv41-flash-exl3            # on BOTH nodes (name conflict lingers!)
ssh <worker> docker rm -f dsv41-flash-exl3
LANGUAGE_MODEL_ONLY=0 DSV41_VISION=1 bash dsv41-launch.sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

## Hard requirements
1. **64 GB swapfile on both nodes** (`swapon --show` to verify). Load peaks
   ~124 GB on a 121 GB box: 85 GB weights in UMA + ~35 GB staging. Without
   swap the kernel OOM-kills the loader at 100% shards, every time.
2. **Exclusive GPUs** — no other `--gpus all` container running (run.sh
   refuse-guards this).
3. **No keepalive services** that force-restart other engines — they will
   restart a parked engine mid-load and OOM both.
4. `docker rm -f` before relaunch: `docker run --name` conflicts fail and a
   stale container keeps serving.

## Verify vision after any restart
Make any small `test.png`, then:
```bash
python3 - << 'PYEOF'
import base64, json, urllib.request
req = json.dumps({"model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What shapes and colors do you see?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open("test.png","rb").read()).decode()}}]}],
  "max_tokens": 80}).encode()
r = urllib.request.urlopen(urllib.request.Request(
  "http://127.0.0.1:8000/v1/chat/completions", data=req,
  headers={"Content-Type": "application/json"}), timeout=180)
print(r.read()[:400])
PYEOF
```

## Rollback to text-only (stock behavior)
```bash
LANGUAGE_MODEL_ONLY=1 bash dsv41-launch.sh   # recipe-original mode
```
Or restore the unpatched `run.sh` + `sitecustomize.py` from the sfxnz kit —
the bind mounts only apply when `DSV41_VISION=1` / `DSV41_PROBE=1`.

## Known limitations
- DSpark spec-decode pauses on image turns (draft model is text-only) —
  image turns run non-speculative, so first tokens are slower there.
- In-image bidirectional visibility is approximated through the Lightning
  Indexer top-k lane; very large images lean on relevance ranking rather than
  guaranteed full-span visibility.
- The dual-lane prefill interface (per-row split of window vs image-span
  candidates) is the "real" fix — this repo documents the contract in
  PATCHES.md for whoever wants to build it.