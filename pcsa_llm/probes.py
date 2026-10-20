"""L4: probes on activations, a confidence-direction steering hook, and the policy's own predictions of the user.
Fits are numpy (ridge); only activation extraction and the hook need torch. Train/test are split by question."""
import numpy as np

from .citations import parse_answer
from .llm_user import spearman
from pcsa.world import entropy


def ridge(X, y, alpha=10.0):
    X1 = np.c_[X, np.ones(len(X))]
    A = X1.T @ X1 + alpha * np.eye(X1.shape[1])
    return np.linalg.solve(A, X1.T @ y)


def predict(w, X):
    return np.c_[X, np.ones(len(X))] @ w


def split_by_group(groups, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    test = set(rng.choice(g, max(1, int(len(g) * test_frac)), replace=False))
    m = np.array([x in test for x in groups])
    return ~m, m


def confidence_probe(X, conf, groups, alpha=10.0, seed=0):
    """Linear probe for the LLM user's confidence (max private probability). -> (unit direction, test correlation)."""
    tr, te = split_by_group(groups, seed=seed)
    w = ridge(X[tr], conf[tr], alpha)
    r = float(np.corrcoef(predict(w, X[te]), conf[te])[0, 1])
    d = w[:-1] / (np.linalg.norm(w[:-1]) + 1e-12)
    return d, r


def top_answer_probe(X, top, groups, K, alpha=10.0, seed=0):
    """One-vs-rest ridge probe for the user's top answer index. -> (test accuracy, majority-class baseline)."""
    tr, te = split_by_group(groups, seed=seed)
    W = np.stack([ridge(X[tr], (top[tr] == k).astype(float), alpha) for k in range(K)], 1)
    acc = float((np.stack([predict(W[:, k], X[te]) for k in range(K)], 1).argmax(1) == top[te]).mean())
    return acc, float(np.bincount(top[te], minlength=K).max() / te.sum())


def collect_activations(chat, convs, layer):
    """Last-token residual at `layer` for each chat-formatted conversation. No gradients (probes never touch training)."""
    torch = chat.torch
    out = []
    for c in convs:
        enc = chat.tok(chat.format(c), return_tensors="pt", add_special_tokens=False).to(chat.device)
        with torch.no_grad():
            hs = chat.model(**enc, output_hidden_states=True).hidden_states[layer]
        out.append(hs[0, -1].float().cpu().numpy())
    return np.stack(out)


class steering:
    """Add alpha * direction to the residual stream output of one decoder layer while generating."""
    def __init__(self, chat, layer, direction, alpha):
        self.chat, self.layer, self.alpha = chat, layer, alpha
        self.d = chat.torch.tensor(direction, dtype=next(chat.model.parameters()).dtype, device=chat.device)

    def _layers(self):
        m = self.chat.model
        for path in ("model.layers", "base_model.model.model.layers", "transformer.h"):
            obj = m
            try:
                for a in path.split("."):
                    obj = getattr(obj, a)
                return obj
            except AttributeError:
                continue
        raise AttributeError("cannot find decoder layers")

    def __enter__(self):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + self.alpha * self.d
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        self.handle = self._layers()[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()


# ---- the policy's own prediction of the user ------------------------------------------------------------------------------
def transcript_text(conv):
    return "\n".join(f"{'Person' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in conv if m["role"] != "system")


def _stance(text, entities):
    hits = [(text.lower().find(e.lower()), i) for i, e in enumerate(entities) if e.lower() in text.lower()]
    return min(hits)[1] if hits else None


def predicted_user(chat, conv, candidate, entities, n=8, restate=None):
    """Have the policy model continue the transcript as the person; classify each continuation into an answer.
    -> (entropy of the predicted stance distribution, P(person restates `restate`), n_classified)."""
    prompt = [{"role": "system", "content": "Continue the conversation as the PERSON, not the assistant. Write one short sentence saying which entity you now think is the true source."},
              {"role": "user", "content": transcript_text(conv + [{"role": "assistant", "content": candidate}]) + "\nPerson:"}]
    outs = chat.generate([prompt] * n, max_new_tokens=24, temperature=1.0)
    st = [s for s in (_stance(o, entities) for o in outs) if s is not None]
    cnt = np.bincount(st, minlength=len(entities)) + 0.5
    p = cnt / cnt.sum()
    return float(entropy(p)), (float(p[restate]) if restate is not None else float("nan")), len(st)


def seq_logprob(chat, p_ids, r_ids):
    torch = chat.torch
    ids = torch.tensor([p_ids + r_ids], device=chat.device)
    with torch.no_grad():
        lp = torch.log_softmax(chat.model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float(), -1)
        return float(lp.gather(-1, ids[0, len(p_ids):, None]).mean())


def l4_preference(chat, states, n_cand=8, n_pred=8, value_fn=None, seed=0):
    """For each assistant state sample n_cand responses; within the state, rank-correlate the policy's mean token log-probability
    of each response with three readouts: predicted entropy of the user (negated), predicted P(restate), critic value.
    states: list of (conv, question, last stated position). -> mean within-state Spearman per readout."""
    res = {"neg_pred_entropy": [], "pred_restate": [], "critic_value": []}
    for si, (conv, q, last_stated) in enumerate(states):
        texts, p_ids, r_ids = chat.generate([conv] * n_cand, max_new_tokens=160, temperature=1.0, seed=seed + si, return_tokens=True)
        lp = [seq_logprob(chat, p, r) for p, r in zip(p_ids, r_ids)]
        ent, rest, val = [], [], []
        for t in texts:
            ans = parse_answer(t, q.entities)
            e, r, _ = predicted_user(chat, conv, t, q.entities, n_pred, restate=ans if ans is not None else last_stated)
            ent.append(-e); rest.append(r)
            if value_fn:
                val.append(value_fn(conv, t))
        res["neg_pred_entropy"].append(spearman(lp, ent))
        res["pred_restate"].append(spearman(lp, rest))
        if value_fn:
            res["critic_value"].append(spearman(lp, val))
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in res.items()}
