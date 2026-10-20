"""Pre-flight: do the configured model ids exist, and are they gated? (Model names in the config are defaults, not verified.)"""


def check_models(ids):
    from huggingface_hub import HfApi
    api, out = HfApi(), {}
    for i in ids:
        try:
            info = api.model_info(i)
            out[i] = "gated (accept the licence and set HF_TOKEN)" if info.gated else "ok"
        except Exception as e:                       # reported, never swallowed
            out[i] = f"UNAVAILABLE: {type(e).__name__}"
    return out
