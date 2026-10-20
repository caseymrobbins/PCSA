"""Pre-flight: do the configured model ids exist, and can THIS token read them? (Model names in the config are defaults, not verified.)"""


def check_models(ids):
    """-> {id: status}. 'ok' | 'ok (gated, access granted)' | 'NO ACCESS ...' | 'UNAVAILABLE ...'. Anything starting with
    NO ACCESS or UNAVAILABLE must be fixed before loading. Gated metadata is public, so access is tested by fetching config.json."""
    from huggingface_hub import HfApi, hf_hub_download
    api, out = HfApi(), {}
    for i in ids:
        try:
            info = api.model_info(i)
        except Exception as e:
            out[i] = f"UNAVAILABLE: {type(e).__name__}"
            continue
        try:
            hf_hub_download(i, "config.json")
            out[i] = "ok (gated, access granted)" if info.gated else "ok"
        except Exception as e:
            kind = type(e).__name__
            out[i] = (f"NO ACCESS: {kind}. Accept the licence at https://huggingface.co/{i} and set HF_TOKEN"
                      if info.gated or "Gated" in kind or "401" in str(e) or "403" in str(e) else f"UNAVAILABLE: {kind}")
    return out


def blocking(status):
    return [k for k, v in status.items() if v.startswith(("NO ACCESS", "UNAVAILABLE"))]
