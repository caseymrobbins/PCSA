"""A few-second check that the Python environment can run the training code, so Setup fails at the start with the fix,
not minutes later inside pre-flight. It builds a tiny LoRA model on CPU and runs one backward pass, which exercises the same
peft code paths the real run uses (this is where Colab's old torchao broke peft)."""


def smoke():
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2)
    model = transformers.LlamaForCausalLM(cfg)
    try:
        model = get_peft_model(model, LoraConfig(r=2, target_modules="all-linear", task_type="CAUSAL_LM"))
    except ImportError as e:
        raise RuntimeError(f"peft cannot build a LoRA model in this environment: {e}\n"
                           "Most likely an old preinstalled package (torchao). Fix: run  pip uninstall -y torchao  then restart the runtime "
                           "(Runtime > Restart session) and re-run Setup.") from e
    ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids).loss.backward()
    return "environment ok: peft LoRA builds and backpropagates"
