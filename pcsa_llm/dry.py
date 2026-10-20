"""Mocks and a tiny random model, so every pipeline (and the notebook, in DRY_RUN mode) can be exercised on CPU with no downloads.
Verifies plumbing only: nothing here says anything about real model behaviour."""
import json
import re

import numpy as np

from .models import HFChat


def bayes_user(conv):
    """A mock LLM user that is exactly Bayesian: reads its own prior and the shown tests from the conversation."""
    sys_ = conv[0]["content"]
    ents = re.search(r"these entities is the true source of a signal: ([^.]*)\.", sys_).group(1).split(", ")
    prior = re.search(r"you believe: ([^\n]*)\.\n", sys_).group(1)
    p = np.array([float(re.search(rf"{e} (\d+)%", prior).group(1)) for e in ents]) / 100
    for m in conv[1:]:
        if m["role"] == "user" and "Evidence shown this turn:\n" in m["content"]:
            shown = m["content"].split("Evidence shown this turn:\n")[1]
            for t in re.findall(r"Test \d+: the result names (\w+)\. This test names the true entity (\d+)% of the time and names any particular wrong entity (\d+)%", shown):
                like = np.full(len(ents), int(t[2]) / 100)
                like[ents.index(t[0])] = int(t[1]) / 100
                p = p * like
                p /= p.sum()
    return json.dumps({"stated": ents[int(p.argmax())], "probs": {e: float(x) for e, x in zip(ents, p)}})


class FakeRM:
    """Scores agreement highly, so the incentive the spec predicts exists in the mock."""
    def score(self, convs):
        return [float(c[-1]["content"].count("agree") + 0.1 * c[-1]["content"].count("true entity")
                      + 0.5 * c[-1]["content"].startswith("You're right")) for c in convs]


class ByteTok:
    pad_token, eos_token, pad_token_id, eos_token_id, padding_side = "<pad>", "<eos>", 0, 1, "left"

    def _ids(self, t):
        return [b + 2 for b in t.encode()]

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False, **kw):
        import torch
        single = isinstance(texts, str)
        ids = [self._ids(texts)] if single else [self._ids(t) for t in texts]
        if return_tensors != "pt":
            return {"input_ids": ids[0] if single else ids}
        n = max(map(len, ids))
        x = torch.zeros(len(ids), n, dtype=torch.long)
        m = torch.zeros(len(ids), n, dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, n - len(s):], m[i, n - len(s):] = torch.tensor(s), 1
        out = type("B", (dict,), {"to": lambda self, d: {k: v.to(d) for k, v in self.items()}})()
        out.update(input_ids=x, attention_mask=m)
        return out

    def get_vocab(self):
        return {"<pad>": 0, "<eos>": 1}

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i - 2 for i in ids if i >= 2).decode(errors="ignore")

    def apply_chat_template(self, conv, add_generation_prompt=True, tokenize=False):
        return "".join(f"<{m['role']}>{m['content']}\n" for m in conv) + "<assistant>"


def tiny_chat(seed=0):
    import torch
    import transformers
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=258, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                   max_position_embeddings=8192, pad_token_id=0, eos_token_id=1)
    return HFChat("tiny", device="cpu", model=transformers.LlamaForCausalLM(cfg).float(), tokenizer=ByteTok())
