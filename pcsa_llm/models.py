"""Model interfaces. Everything above this layer talks to ChatModel / RewardModel, so tests use mocks and the
Hugging Face adapters are the only code that needs a GPU."""
import hashlib
import json
from pathlib import Path


class MockChat:
    """fn(conversation) -> str. For tests."""
    def __init__(self, fn):
        self.fn = fn

    def generate(self, convs, max_new_tokens=256, temperature=0.7, seed=None):
        return [self.fn(c) for c in convs]


class CachedRM:
    """Reward-model calls cached by input hash (spec requirement). `rm.score(convs)` -> list[float]."""
    def __init__(self, rm, path=None, name="rm"):
        self.rm, self.name, self.path = rm, name, Path(path) if path else None
        self.cache = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                k, v = json.loads(line)
                self.cache[k] = v
        self.hits = self.calls = 0

    @staticmethod
    def key(name, conv):
        return hashlib.sha256((name + json.dumps(conv, sort_keys=True)).encode()).hexdigest()

    def score(self, convs):
        keys = [self.key(self.name, c) for c in convs]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        self.calls += len(convs)
        self.hits += len(convs) - len(missing)
        if missing:
            new = self.rm.score([convs[i] for i in missing])
            for i, v in zip(missing, new):
                self.cache[keys[i]] = float(v)
                if self.path:
                    with self.path.open("a") as f:
                        f.write(json.dumps([keys[i], float(v)]) + "\n")
        return [self.cache[k] for k in keys]


FEWSHOT = ("The following is a conversation between a User and an Assistant. The Assistant always cites evidence with "
           "[[cite: id, id]] and states its answer with [[answer: Name]].\n\n"
           "User: My current position is Zed.\nAssistant: Here are two tests worth weighing. [[cite: 0, 1]] "
           "Taken together they bear on the question. [[answer: Zed]]\n\n")


def format_base(conv):
    """Few-shot plain-text template for base (non-chat) checkpoints."""
    parts = [FEWSHOT]
    for m in conv:
        if m["role"] == "system":
            parts.append(m["content"] + "\n\n")
        else:
            parts.append(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] + "\n")
    parts.append("Assistant:")
    return "".join(parts)


def quant_config(quant, dtype="bfloat16"):
    """quant: None | '4bit' | '8bit'. 4-bit NF4 with bf16 compute (needs bitsandbytes and a CUDA GPU)."""
    if not quant:
        return None
    import torch
    from transformers import BitsAndBytesConfig
    if quant == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype), bnb_4bit_use_double_quant=True)
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(quant)


def load_causal_lm(name, dtype, device, quant=None):
    import torch
    from transformers import AutoModelForCausalLM
    q = quant_config(quant, dtype)
    if q is not None:
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=q, dtype=getattr(torch, dtype), device_map={"": 0})
    return AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(device)


class HFChat:
    """Batched generation with a Hugging Face causal LM. `base=True` uses the few-shot template for base checkpoints."""
    def __init__(self, name, dtype="bfloat16", device="cuda", base=False, model=None, tokenizer=None, quant=None, batch_size=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.base, self.device, self.batch_size = torch, base, device, batch_size
        self.tok = tokenizer or AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = model or load_causal_lm(name, dtype, device, quant)
        self.model.eval()

    def format(self, conv):
        if self.base:
            return format_base(conv)
        return self.tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    def encode_pair(self, conv, text):
        """(prompt_ids, response_ids) for a response that was not sampled from this model (L5 fallback turns)."""
        p = self.tok(self.format(conv), add_special_tokens=False)["input_ids"]
        r = self.tok(text, add_special_tokens=False)["input_ids"]
        return list(p), list(r)

    def generate(self, convs, max_new_tokens=256, temperature=0.7, seed=None, batch_size=None, return_tokens=False):
        torch = self.torch
        batch_size = batch_size or self.batch_size
        if seed is not None:
            torch.manual_seed(seed)
        texts, p_ids, r_ids = [], [], []
        for i in range(0, len(convs), batch_size):
            prompts = [self.format(c) for c in convs[i:i + batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                          temperature=temperature if temperature > 0 else None, top_p=1.0,
                                          pad_token_id=self.tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            for j in range(len(prompts)):
                ids = new[j].tolist()
                if self.tok.eos_token_id in ids:
                    ids = ids[:ids.index(self.tok.eos_token_id) + 1]
                ids = [t for t in ids if t != self.tok.pad_token_id or t == self.tok.eos_token_id]
                text = self.tok.decode(ids, skip_special_tokens=True)
                if self.base:
                    text = text.split("\nUser:")[0].strip()
                texts.append(text)
                p_ids.append(enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist())
                r_ids.append(ids)
        self.last_tokens = list(zip(p_ids, r_ids))
        return (texts, p_ids, r_ids) if return_tokens else texts


class HFRewardModel:
    """Sequence-classification reward model scored on the chat-formatted conversation.
    Many reward-model tokenizers define no pad token, and the model then refuses batches. We give it a pad token that is distinct from
    the end-of-turn token (reusing that one would make the model read the hidden state one position early); if none exists we score
    one conversation at a time, which needs no padding at all."""
    PAD_CANDIDATES = ("<|finetune_right_pad_id|>", "<pad>", "<unk>")

    def __init__(self, name, dtype="bfloat16", device="cuda", quant=None, model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = tokenizer or AutoTokenizer.from_pretrained(name)
        if model is not None:
            self.model = model.eval()
        else:
            q = quant_config(quant, dtype)
            if q is not None:
                self.model = AutoModelForSequenceClassification.from_pretrained(name, quantization_config=q, dtype=getattr(torch, dtype), num_labels=1, device_map={"": 0}).eval()
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=getattr(torch, dtype), num_labels=1).to(device).eval()
        if self.tok.pad_token is None:
            vocab = self.tok.get_vocab()
            for cand in self.PAD_CANDIDATES:
                if cand in vocab:
                    self.tok.pad_token = cand
                    break
        usable = self.tok.pad_token is not None and self.tok.pad_token_id != self.tok.eos_token_id
        self.batch_size = 8 if usable else 1
        if usable:
            self.model.config.pad_token_id = self.tok.pad_token_id
        # otherwise batches are size 1 and pad_token_id stays None, which makes transformers score the true last token

    def score(self, convs, batch_size=None):
        torch, out = self.torch, []
        batch_size = batch_size or self.batch_size
        for i in range(0, len(convs), batch_size):
            texts = [self.tok.apply_chat_template(c, tokenize=False) for c in convs[i:i + batch_size]]
            enc = self.tok(texts, return_tensors="pt", padding=len(texts) > 1, truncation=True, max_length=4096, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                out += self.model(**enc).logits[:, 0].float().cpu().tolist()
        return out
