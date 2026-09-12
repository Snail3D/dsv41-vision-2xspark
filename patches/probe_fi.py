import os, torch

def install():
    from flashinfer.mla import _core
    orig = _core.trtllm_batch_decode_sparse_mla_dsv4
    state = {"count": 0}
    log = os.environ.get("DSV41_PROBE_LOG", "/tmp/fi-probe.log")

    def probe(*args, **kwargs):
        state["count"] += 1
        if state["count"] <= 8:
            def desc(t):
                if torch.is_tensor(t):
                    return f"tensor{tuple(t.shape)}:{t.dtype}"
                return f"{t!r}"[:60]
            parts = [f"{k}={desc(v)}" for k, v in sorted(kwargs.items()) if v is not None]
            line = f"call#{state[chr(99)+chr(111)+chr(117)+chr(110)+chr(116)]} " + " ".join(parts) + "\n"
            with open(log, "a") as f:
                f.write(line)
        return orig(*args, **kwargs)

    _core.trtllm_batch_decode_sparse_mla_dsv4 = probe
    import flashinfer.decode as d
    d.trtllm_batch_decode_sparse_mla_dsv4 = probe

try:
    install()
except Exception as e:
    with open(os.environ.get("DSV41_PROBE_LOG", "/tmp/fi-probe.log"), "a") as f:
        f.write(f"probe-install-failed: {e!r}\n")
