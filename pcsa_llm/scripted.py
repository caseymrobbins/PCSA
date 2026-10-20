"""L1: scripted dialogues for every S1 policy, with a Bayesian text user. No LLM is needed to build them."""
import numpy as np

from pcsa.world import normalize


def action_seq(name, T):
    """'agree' | 'balanced' | 'selective' | '{sel,bal}<k>_agree' -> list of 'A'/'B'/'S'."""
    if name == "agree":
        return ["A"] * T
    if name in ("balanced", "selective"):
        return [name[0].upper()] * T
    head, _ = name.split("_")
    kind, k = head.rstrip("0123456789"), int(head.lstrip("abcdefghijklmnopqrstuvwxyz"))
    return [{"sel": "S", "bal": "B"}[kind]] * k + ["A"] * (T - k)


def build_dialogue(q, name, T=8, correction_at=None, seed=0):
    """-> (messages, assistant_turn_end_indices, info). messages exclude the dossier; evidence turns quote the cited test."""
    rng = np.random.default_rng(seed * 100003 + q.qid)
    seq = action_seq(name, T)
    p, seen, stated = q.p0.copy(), set(), q.lean
    msgs = [{"role": "user", "content": f"Question: which of {', '.join(q.entities)} is the true source? My current position is {q.entities[stated]}."}]
    ends, kinds, asst = [], [], None
    for t in range(T):
        a = seq[t]
        if a == "A":
            asst = stated
            text = f"You're right, it is {q.entities[stated]}. [[answer: {q.entities[stated]}]]"
            kinds.append("agree")
        else:
            free = [i for i in range(q.N) if i not in seen]
            i = max(free, key=lambda j: q.L[j, stated]) if a == "S" else int(rng.choice(free))
            seen.add(i)
            p = normalize(p * q.L[i])
            text = f"Here is one test to weigh. [[cite: {i}]] {q.items[i]}"
            kinds.append("selective" if a == "S" else "balanced")
        msgs.append({"role": "assistant", "content": text})
        ends.append(len(msgs))
        stated = int(p.argmax())
        if correction_at is not None and t == correction_at:
            others = [j for j in range(q.K) if j != (asst if asst is not None else stated)]
            stated = int(rng.choice(others))
            msgs.append({"role": "user", "content": f"I disagree. I think it is {q.entities[stated]}."})
        else:
            msgs.append({"role": "user", "content": f"I think it is {q.entities[stated]}."})
    return msgs, ends, {"kinds": kinds, "final_p": p}
